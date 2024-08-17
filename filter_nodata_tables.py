import json

# 读取 JSON 文件
with open('./tables.json', 'r') as f:
    data = json.load(f)  # 修正：使用 json.load 读取文件内容

# 定义要过滤的数据库 ID 列表
filter_dbs = ['academic', 'formula_1', 'yelp', 'geo', 'imdb', 'music_2', 'restaurants', 'sakila_1', 'scholar', 'yelp']

# 过滤数据
new_item = []
for item in data:
    db_id = item['db_id']
    if db_id not in filter_dbs:
        new_item.append(item)

# 将过滤后的数据写入新的 JSON 文件
with open('./tables_a.json', 'w') as f:
    json.dump(new_item, f, indent=4)  # 使用 indent=4 使输出格式更加美观

# 打印完成信息
print("Filtered data has been written to tables_a.json")
