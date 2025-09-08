import os
import json
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count

def process_movement_sequence_file(args):
    data_folder, filename, output_dir, save_format, map_filter = args
    output_data = {}
    round_end_reasons = defaultdict(int)
    hit_group_counts = defaultdict(int)
    try:
        with open(os.path.join(data_folder, filename), 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return filename, defaultdict(int), defaultdict(int)

    online, file_id = filename.split('\\')[-1].split('.')[0].split('_')
    map_name = data["mapName"]

    # Skip file if it doesn't match the map filter
    if map_filter and map_name != map_filter:
        return filename, defaultdict(int), defaultdict(int)

    for round_idx, round_data in enumerate(data['gameRounds']):
        try:
            # Get round outcome data
            round_end_reason = round_data["roundEndReason"]
            round_end_reasons[round_end_reason] += 1
            winning_side = round_data["winningSide"]

            if round_end_reason == "GameStart":
                continue

            t_player_trajectory = defaultdict(list)
            ct_player_trajectory = defaultdict(list)
            t_player_hp_timeseries = defaultdict(list)
            ct_player_hp_timeseries = defaultdict(list)
            t_player_seq_len = defaultdict(int)
            ct_player_seq_len = defaultdict(int)
            t_player_armor = {}
            ct_player_armor = {}
            t_player_helmet = {}
            ct_player_helmet = {}
            t_player_weapons_timeseries = defaultdict(list)
            ct_player_weapons_timeseries = defaultdict(list)
            t_player_view_x = defaultdict(list)
            ct_player_view_x = defaultdict(list)
            bomb_trajectory = []
            last_bomb_location = None
            damage_outcomes = defaultdict(list)

            # Get player lists first to ensure consistent ordering
            t_players = []
            ct_players = []
            if "players" in round_data["tSide"] and "players" in round_data["ctSide"] and round_data["tSide"]["players"] and round_data["ctSide"]["players"]:
                t_players = [p["steamID"] for p in round_data["tSide"]["players"]]
                ct_players = [p["steamID"] for p in round_data["ctSide"]["players"]]
            else:
                for player in round_data["frames"][2]["t"]["players"]:
                    t_players.append(player["steamID"])
                for player in round_data["frames"][2]["ct"]["players"]:
                    ct_players.append(player["steamID"])
            assert len(t_players) == 5 and len(ct_players) == 5

            # Get initial states from first frame
            if len(round_data["frames"]) > 0:
                for steam_id in t_players:
                    t_player_weapons_timeseries[steam_id] = []
                for steam_id in ct_players:
                    ct_player_weapons_timeseries[steam_id] = []

                frame_idx = 0
                missing_weapons = True
                
                # Continue until all players have weapons or we run out of frames
                while missing_weapons and frame_idx < len(round_data["frames"]):
                    current_frame = round_data["frames"][frame_idx]
                    for side in ["t", "ct"]:
                        player_list = t_players if side == "t" else ct_players
                        player_dict = t_player_weapons_timeseries if side == "t" else ct_player_weapons_timeseries
                        
                        if side in current_frame and "players" in current_frame[side]:
                            # Create a mapping of steamID to player data
                            player_data = {p["steamID"]: p for p in current_frame[side]["players"]}
                            
                            # Process players in the correct order
                            for steam_id in player_list:
                                player = player_data.get(steam_id, {})
                                # Get armor and helmet status (only from first frame)
                                if frame_idx == 0:
                                    t_player_armor[steam_id] = player.get('armor', 0)
                                    t_player_helmet[steam_id] = player.get('hasHelmet', False)
                                    ct_player_armor[steam_id] = player.get('armor', 0)
                                    ct_player_helmet[steam_id] = player.get('hasHelmet', False)
                                
                                # Only update weapon if player doesn't have one yet
                                if player_dict[steam_id] == "None":
                                    # Get weapon
                                    main_weapon = "None"
                                    pistol = "None"
                                    if "inventory" in player and player["inventory"] is not None:
                                        for item in player["inventory"]:
                                            weapon_class = item.get("weaponClass", "")
                                            if weapon_class in ["Rifle", "SMG", "Heavy"]:
                                                main_weapon = item["weaponName"]
                                            elif weapon_class == "Pistols":
                                                pistol = item["weaponName"]
                                        if main_weapon != "None" or pistol != "None":
                                            player_dict[steam_id] = main_weapon if main_weapon != "None" else pistol
                    
                    # Check if all players have weapons
                    missing_weapons = any(weapon == "None" for weapon in t_player_weapons_timeseries.values()) or \
                                    any(weapon == "None" for weapon in ct_player_weapons_timeseries.values())
                    frame_idx += 1

                if missing_weapons:
                    print(f"Warning: Could not find weapons for all players in {filename}, round {round_idx}")

            last_t_locations = {}
            last_ct_locations = {}
            last_t_hp = {}
            last_ct_hp = {}

            # Find initial bomb carrier by checking first 5 frames
            initial_bomb_carrier_player_idx = None
            initial_bomb_location = None
            for frame_idx in range(min(5, len(round_data["frames"]))):
                frame = round_data["frames"][frame_idx]
                if initial_bomb_location is None and 'bomb' in frame and all(coord in frame['bomb'] for coord in ['x', 'y', 'z']):
                    initial_bomb_location = [frame['bomb']['x'], frame['bomb']['y'], frame['bomb']['z']]
                
                if "t" in frame and "players" in frame["t"]:
                    for idx, player in enumerate(frame["t"]["players"]):
                        if player.get("hasBomb", False):
                            initial_bomb_carrier_player_idx = idx
                            break
                    if initial_bomb_carrier_player_idx is not None:
                        break

            # If we never found a bomb location, set default
            if initial_bomb_location is None:
                print(f"No bomb location found in {filename}, round {round_idx}")

            # put damages into bins
            starting_tick = round_data["frames"][0]["tick"]
            bin_tick_size = 64
            for damage in round_data["damages"]:
                if damage['weaponClass'] in ["Equipment", "Grenade"]:
                    continue
                if damage["isFriendlyFire"]:
                    continue
                hit_group = damage.get("hitGroup", "Generic")
                if hit_group == "Generic":
                    continue
                if hit_group in ["LeftArm", "RightArm"]:
                    hit_group = "Arm"
                if hit_group in ["LeftLeg", "RightLeg"]:
                    hit_group = "Leg"
                damage_tick = damage["tick"] - starting_tick
                damage_value = damage["hpDamage"]
                attacker_steam_id = damage["attackerSteamID"]
                victim_steam_id = damage["victimSteamID"]
                
                hit_group_counts[hit_group] += 1
                attacker_player_idx = t_players.index(attacker_steam_id) if attacker_steam_id in t_players else ct_players.index(attacker_steam_id) + len(t_players)
                victim_player_idx = t_players.index(victim_steam_id) if victim_steam_id in t_players else ct_players.index(victim_steam_id) + len(t_players)
                damage_event = (attacker_player_idx, victim_player_idx, damage_value, hit_group)
                bin_idx = damage_tick // bin_tick_size
                damage_outcomes[bin_idx].append(damage_event)

            # collect player trajectories
            for idx, frame in enumerate(round_data["frames"]):
                if 'bomb' in frame and all(coord in frame['bomb'] for coord in ['x', 'y', 'z']):
                    bomb_pos = [frame['bomb']['x'], frame['bomb']['y'], frame['bomb']['z']]
                    last_bomb_location = bomb_pos
                else:
                    bomb_pos = last_bomb_location if last_bomb_location else [0, 0, 0]
                bomb_trajectory.append(bomb_pos)

                t_ids_in_frame = set()
                if frame["t"]["players"]:
                    t_ids_in_frame = {player["steamID"] for player in frame["t"]["players"]}
                    for player in frame["t"]["players"]:
                        if player['isAlive']:
                            t_player_seq_len[player["steamID"]] += 1
                        t_player_trajectory[player["steamID"]].append([player['x'], player['y'], player['z']])
                        t_player_hp_timeseries[player["steamID"]].append(player['hp'])
                        t_player_view_x[player["steamID"]].append(player['viewX'])
                        # Add weapon tracking
                        main_weapon = "None"
                        pistol = "None"
                        if "inventory" in player and player["inventory"] is not None:
                            for item in player["inventory"]:
                                weapon_class = item.get("weaponClass", "")
                                if weapon_class in ["Rifle", "SMG", "Heavy"]:
                                    main_weapon = item["weaponName"]
                                elif weapon_class == "Pistols":
                                    pistol = item["weaponName"]
                        t_player_weapons_timeseries[player["steamID"]].append(main_weapon if main_weapon != "None" else pistol)
                        
                        last_t_locations[player["steamID"]] = [player['x'], player['y'], player['z']]
                        last_t_hp[player["steamID"]] = player['hp']
                for t_player in t_players:
                    if t_player not in t_ids_in_frame:
                        if t_player in last_t_locations:
                            t_player_trajectory[t_player].append(last_t_locations[t_player])
                            t_player_hp_timeseries[t_player].append(last_t_hp.get(t_player, 0))
                            # Add view_x handling for missing players
                            last_view_x = t_player_view_x[t_player][-1] if t_player_view_x[t_player] else 0
                            t_player_view_x[t_player].append(last_view_x)
                            # Use last known weapon or "None" if no weapon history
                            last_weapon = t_player_weapons_timeseries[t_player][-1] if t_player_weapons_timeseries[t_player] else "None"
                            t_player_weapons_timeseries[t_player].append(last_weapon)

                ct_ids_in_frame = set()
                if frame["ct"]["players"]:
                    ct_ids_in_frame = {player["steamID"] for player in frame["ct"]["players"]}
                    for player in frame["ct"]["players"]:
                        if player['isAlive']:
                            ct_player_seq_len[player["steamID"]] += 1
                        ct_player_trajectory[player["steamID"]].append([player['x'], player['y'], player['z']])
                        ct_player_hp_timeseries[player["steamID"]].append(player['hp'])
                        ct_player_view_x[player["steamID"]].append(player['viewX'])
                        # Add weapon tracking
                        main_weapon = "None"
                        pistol = "None"
                        if "inventory" in player and player["inventory"] is not None:
                            for item in player["inventory"]:
                                weapon_class = item.get("weaponClass", "")
                                if weapon_class in ["Rifle", "SMG", "Heavy"]:
                                    main_weapon = item["weaponName"]
                                elif weapon_class == "Pistols":
                                    pistol = item["weaponName"]
                        ct_player_weapons_timeseries[player["steamID"]].append(main_weapon if main_weapon != "None" else pistol)
                        
                        last_ct_locations[player["steamID"]] = [player['x'], player['y'], player['z']]
                        last_ct_hp[player["steamID"]] = player['hp']
                for ct_player in ct_players:
                    if ct_player not in ct_ids_in_frame:
                        if ct_player in last_ct_locations:
                            ct_player_trajectory[ct_player].append(last_ct_locations[ct_player])
                            ct_player_hp_timeseries[ct_player].append(last_ct_hp.get(ct_player, 0))
                            # Add view_x handling for missing players
                            last_view_x = ct_player_view_x[ct_player][-1] if ct_player_view_x[ct_player] else 0
                            ct_player_view_x[ct_player].append(last_view_x)
                            # Use last known weapon or "None" if no weapon history
                            last_weapon = ct_player_weapons_timeseries[ct_player][-1] if ct_player_weapons_timeseries[ct_player] else "None"
                            ct_player_weapons_timeseries[ct_player].append(last_weapon)

                player_trajectory = []
                player_hp_timeseries = []
                player_seq_len = []
                player_view_x = []
                for player in t_players:
                    player_trajectory.append(t_player_trajectory[player])
                    player_hp_timeseries.append(t_player_hp_timeseries[player])
                    player_seq_len.append(t_player_seq_len[player])
                    player_view_x.append(t_player_view_x[player])
                for player in ct_players:
                    player_trajectory.append(ct_player_trajectory[player])
                    player_hp_timeseries.append(ct_player_hp_timeseries[player])
                    player_seq_len.append(ct_player_seq_len[player])
                    player_view_x.append(ct_player_view_x[player])

            # Convert to numpy arrays AFTER the frame loop
            player_trajectory = np.array(player_trajectory, dtype=np.float32)
            player_hp_timeseries = np.array(player_hp_timeseries, dtype=np.float32)
            player_view_x = np.array(player_view_x, dtype=np.float32)
            bomb_trajectory = np.array(bomb_trajectory, dtype=np.float32)

            # Combine player states in the same order as timeseries
            player_armor = [t_player_armor.get(p, 0) for p in t_players] + [ct_player_armor.get(p, 0) for p in ct_players]
            player_helmet = [t_player_helmet.get(p, False) for p in t_players] + [ct_player_helmet.get(p, False) for p in ct_players]
            player_weapons_timeseries = []
            for player in t_players:
                player_weapons_timeseries.append(t_player_weapons_timeseries[player])
            for player in ct_players:
                player_weapons_timeseries.append(ct_player_weapons_timeseries[player])
            player_ids = [f"T_{i}" for i in range(len(t_players))] + [f"CT_{i}" for i in range(len(ct_players))]

            if save_format == 'json':
                output_file_name = f"{online}_{file_id}_{round_idx}.json"
                output_data = {
                    'player_trajectory': player_trajectory.tolist(),
                    'player_hp_timeseries': player_hp_timeseries.tolist(),
                    'player_view_x': player_view_x.tolist(),
                    'player_seq_len': player_seq_len,
                    'player_armor': player_armor,
                    'player_helmet': player_helmet,
                    'player_weapons_timeseries': player_weapons_timeseries,
                    'player_ids': player_ids,
                    'bomb_trajectory': bomb_trajectory.tolist(),
                    'map_name': map_name,
                    'online': online,
                    'round_idx': round_idx,
                    'round_end_reason': round_end_reason,
                    'winning_side': winning_side,
                    'initial_bomb_carrier_player_idx': initial_bomb_carrier_player_idx,
                    'initial_bomb_location': initial_bomb_location,
                    'damage_outcomes': damage_outcomes
                }
                
                with open(f'{output_dir}/{output_file_name}', 'w') as f:
                    json.dump(output_data, f)
            else:  # npz format
                output_file_name = f"{online}_{file_id}_{round_idx}.npz"
                np.savez(f'{output_dir}/{output_file_name}', 
                        player_trajectory=player_trajectory,
                        player_hp_timeseries=player_hp_timeseries,
                        player_view_x=player_view_x,
                        player_seq_len=player_seq_len,
                        player_armor=player_armor,
                        player_helmet=player_helmet,
                        player_weapons_timeseries=player_weapons_timeseries,
                        player_ids=player_ids,
                        bomb_trajectory=bomb_trajectory,
                        map_name=map_name,
                        online=online,
                        round_idx=round_idx,
                        round_end_reason=round_end_reason,
                        winning_side=winning_side,
                        initial_bomb_carrier_player_idx=initial_bomb_carrier_player_idx,
                        initial_bomb_location=initial_bomb_location,
                        damage_outcomes=damage_outcomes)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            print(f"Error occurred at line {e.__traceback__.tb_lineno}")
            continue
            
    return filename, round_end_reasons, hit_group_counts

def build_player_movement_sequence_dataset_parallel(data_folder, output_dir, save_format='json', map_filter=None):
    files = [(data_folder, filename, output_dir, save_format, map_filter) 
             for filename in os.listdir(data_folder)]
    all_round_end_reasons = defaultdict(int)
    all_hit_group_counts = defaultdict(int)
    
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_movement_sequence_file, files), total=len(files)))
    
    # Sum up all counts
    for result in results:
        if result is not None:
            _, round_end_reasons, hit_group_counts = result
            for reason, count in round_end_reasons.items():
                all_round_end_reasons[reason] += count
            for hit_group, count in hit_group_counts.items():
                all_hit_group_counts[hit_group] += count
            
    print("\nRound end reason counts:")
    total_rounds = sum(all_round_end_reasons.values())
    for reason, count in sorted(all_round_end_reasons.items()):
        percentage = (count / total_rounds) * 100
        print(f"- {reason}: {count} ({percentage:.1f}%)")
        
    print("\nHit group counts:")
    total_hits = sum(all_hit_group_counts.values())
    for hit_group, count in sorted(all_hit_group_counts.items()):
        percentage = (count / total_hits) * 100
        print(f"- {hit_group}: {count} ({percentage:.1f}%)")


    
if __name__ == "__main__":
    DATA_FOLDER = '../data'
    EXTRACTED_DATA_FOLDER = os.path.join(DATA_FOLDER, 'extracted_data')
    
    # Modify output folder name based on format and map
    save_format = 'npz'  # or 'npz'
    # map_filter = 'de_dust2'
    map_filter = None
    
    folder_suffix = f'_{map_filter}' if map_filter else '_full'
    PLAYER_SEQ_ALLMAP_FOLDER = os.path.join(DATA_FOLDER, f'player_seq_allmap{folder_suffix}_{save_format}_damage')
    
    if not os.path.exists(PLAYER_SEQ_ALLMAP_FOLDER):
        os.makedirs(PLAYER_SEQ_ALLMAP_FOLDER)

    build_player_movement_sequence_dataset_parallel(
        EXTRACTED_DATA_FOLDER, 
        PLAYER_SEQ_ALLMAP_FOLDER, 
        save_format,
        map_filter
    )

    # Add these lines to process damage data
    # DAMAGE_DATA_FOLDER = os.path.join(DATA_FOLDER, 'damage_data')
    # if not os.path.exists(DAMAGE_DATA_FOLDER):
    #     os.makedirs(DAMAGE_DATA_FOLDER)
    
    # build_damage_dataset_parallel(EXTRACTED_DATA_FOLDER, DAMAGE_DATA_FOLDER)
