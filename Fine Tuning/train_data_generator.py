import pandas as pd

excel_file = 'data.xlsx'
data = pd.read_excel(excel_file)

q = data.iloc[0:107, 1:2]
a = data.iloc[0:107, 2:3]

with open('train_data.jsonl', 'w') as f:
	for i in range(106):
		line = '{"prompt": "' + q.loc[i][0] + '", "completion": "' + a.loc[i][0] + '"}\n'
		f.write(line)