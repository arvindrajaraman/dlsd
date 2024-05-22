achievement_labels = ['collect wood', 'place table', 'eat cow', 'collect sapling',
                        'collect drink', 'make wood pickaxe', 'make wood sword', 'place plant',
                        'defeat zombie', 'collect stone', 'place stone', 'eat plant',
                        'defeat skeleton', 'make stone pickaxe', 'make stone sword', 'wake up',
                        'place furnace', 'collect coal', 'collect iron', 'collect diamond',
                        'make iron pickaxe', 'make iron sword']

action_names = ['no op', 'left', 'right', 'up', 'down', 'do', 'sleep', 'place stone', 'place table',
                'place furnace', 'place plant', 'make wood pickaxe', 'make stone pickaxe',
                'make iron pickaxe', 'make wood sword', 'make stone sword', 'make iron sword']

blocks_labels = ['invalid', 'out of bounds', 'grass', 'water', 'stone', 'tree', 'wood', 'path',
                 'coal', 'iron', 'diamond', 'crafting table', 'furnace', 'sand', 'lava', 'plant',
                 'ripe plant']

mobs_labels = ['zombie', 'cow', 'skeleton', 'arrow']

inventory_labels = ['wood', 'stone', 'coal', 'iron', 'diamond', 'sapling', 'wood pickaxe',
                    'stone pickaxe', 'iron pickaxe', 'wood sword', 'stone sword', 'iron sword']

feature_names = [f'block/{l}' for l in blocks_labels]
feature_names += [f'mob/{l}' for l in mobs_labels]
feature_names += [f'inv/{l}' for l in inventory_labels]
feature_names += ['int/player_health', 'int/player_food', 'int/player_drink', 'int/player_energy',
                  'dir/1', 'dir/2', 'dir/3', 'dir/4', 'light_level', 'is_sleeping']
