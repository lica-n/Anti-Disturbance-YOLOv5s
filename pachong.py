import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://lishi.tianqi.com/hangzhou/201607.html'
response = requests.get(url)
html = response.text

soup = BeautifulSoup(html, 'html.parser')
weather_list = soup.find_all('div', class_='tqtongji2')

data = []  # 用于存储提取到的数据

for weather in weather_list:
    date = weather.find('ul').find_all('li')[0].get_text()
    weather_desc = weather.find('ul').find_all('li')[1].get_text()
    temperature = weather.find('ul').find_all('li')[2].get_text()
    wind = weather.find('ul').find_all('li')[3].get_text()

    # 将提取到的数据存储到列表中
    data.append({'Date': date, 'Weather': weather_desc, 'Temperature': temperature, 'Wind': wind})

# 创建一个 DataFrame 对象
df = pd.DataFrame(data)

# 将数据保存到 Excel 文件
df.to_excel('weather_data.xlsx', index=False)