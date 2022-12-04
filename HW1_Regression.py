import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
import re
from sklearn.preprocessing import StandardScaler


df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')


# Обработка дубликатов

train_without_price = df_train.drop('selling_price', axis=1)  # исключаем целевую переменную

dup = train_without_price.duplicated()  # находим дубликаты
train_without_price = train_without_price.drop_duplicates()  # удаляем дубликаты
df_train = pd.concat([train_without_price, df_train['selling_price']], axis=1, join="inner")  # возвращаем целевую переменную

df_train.reset_index(drop=True, inplace=True) # восстанавливаем порядок индексации


# Обработка столбцов:

  # Обработаем столбец mileage

def fixing_mileage(mileage):
  new_mileage = []

  for mil in mileage:
     if type(mil) == str:  # способ обходить столбец, не трогая пустые значения
       new_mileage.append(float('.'.join(re.findall('\d+', mil))))  # выделяем из строки числовую часть
     else:
       new_mileage.append(mil)  # NaN будем оставлять нетронутыми
  return new_mileage

new_mileage_train = fixing_mileage(df_train['mileage'])
df_train['mileage'] = new_mileage_train

new_mileage_test = fixing_mileage(df_test['mileage'])
df_test['mileage'] = new_mileage_test


  # Обработаем столбец engine

def fixing_engine(engine):
  new_engine = []

  for eng in engine:
     if type(eng) == str:  # способ обходить столбец, не трогая пустые значения
       new_engine.append(float('.'.join(re.findall('\d+', eng))))  # выделяем из строки числовую часть
     else:
       new_engine.append(eng)  # NaN будем оставлять нетронутыми

  return new_engine

new_engine_train = fixing_engine(df_train['engine'])
df_train['engine'] = new_engine_train

new_engine_test = fixing_engine(df_test['engine'])
df_test['engine'] = new_engine_test


  # Обработаем столбец max_power

def fixing_max_power(max_power):
  new_max_power = []

  for power in max_power:
    if type(power) == str:  # способ обходить столбец, не трогая пустые значения
       numb = '.'.join(re.findall('\d+', power))  # выделяем из строки числовую часть
       if numb == '':  # обрабатываем значения без чисел
         new_max_power.append(np.nan)
       elif '.' in numb:  # обрабатываем дробные числа
         new_max_power.append(float(numb))
       else:  # обрабатываем целые числа
         new_max_power.append(float(int(numb)))
    else:  # NaN будем оставоять нетронутыми
       new_max_power.append(power)

  return new_max_power

new_max_power_train = fixing_max_power(df_train['max_power'])
df_train['max_power'] = new_max_power_train

new_max_power_test = fixing_max_power(df_test['max_power'])
df_test['max_power'] = new_max_power_test


  # Обработаем столбец torque

    # Выделяем новый столбец max_torque_rpm

def fixing_max_torque_rpm(max_torque_rpm):
  new_max_torque_rpm = []

  for torq in max_torque_rpm:
    if type(torq) == str:  # способ обходить столбец, не трогая пустые значения
      max_torque = re.findall('[\d+,]{4}[\d+,. +\/-]*', torq)  # находим все max_torque_rpm
      if max_torque == []:  # есть 3 объекта с отсутствующим max_torque_rpm показателем
        curr_max_torque = np.nan
      else:  # сохраняем найденное значение
        curr_max_torque = max_torque[0]
      new_max_torque_rpm.append(curr_max_torque)
    else:  # сохраняем пустые значения (nan)
      new_max_torque_rpm.append(torq)

  processed_max_torque_rpm = []

  for torq in new_max_torque_rpm:
    if type(torq) == str:
      if '-' not in torq and '/' not in torq: # обрабатываем обычные числа
        numb = int(''.join(re.findall('\d+', torq)))  # выделяем из строки числовую часть
      elif '/' in torq: # обрабатываем диапозоны вида +/-
        numbs = re.findall('[\d,]+', torq)  # сохраняем все числа
        numb = int(''.join(re.findall('\d+', numbs[0])))  # первое число будет средним диапазона
      else: # обрабатываем диапазоны
        # разделим диапазоны на два числа
        range_of_numbs = re.findall('[\d+,]+', torq)
        first_numb = int(''.join(re.findall('\d+', range_of_numbs[0])))
        second_numb = int(''.join(re.findall('\d+', range_of_numbs[1])))
        numb = int((first_numb + second_numb) / 2) # сохраним среднее диапазона
      processed_max_torque_rpm.append(numb)
    else:  # сохраняем пустые значения (nan)
      processed_max_torque_rpm.append(torq)
  return processed_max_torque_rpm

    # Перезапишем столбцы для df_train

new_max_torque_rpm_train = fixing_max_torque_rpm(df_train['torque'])
df_train['max_torque_rpm'] = new_max_torque_rpm_train

new_max_torque_rpm_test = fixing_max_torque_rpm(df_test['torque'])
df_test['max_torque_rpm'] = new_max_torque_rpm_test


    # Выделяем новый столбец torque

def fixing_torque(torque):
  new_torque = []

  for torq in torque:
    if type(torq) == str:  # способ обходить столбец, не трогая пустые значения
      torque = re.findall('\d+[.,\d+]*[ Nkg@mn]+', torq)  # находим все torque
      if torque == []:  # для строки '110(11.2)@ 4800'
        torque = re.findall('\d+', torq)
      curr_torque = torque[0]
      if 'Nm' not in curr_torque and 'nm' not in curr_torque and 'kgm' not in curr_torque: # если единицы измерения отсутствуют
        unit = re.findall('[NnKkGg]+[mM]', torq)  # находим единицы измерения
        if unit != []:  # если измерения нашлись
          curr_torque = torque[0] + unit[0] # объединяем измерение и показатель
      new_torque.append(curr_torque)
    else:  # сохраняем пустые значения (nan)
      new_torque.append(torq)

  processed_torque = []

  for torq in new_torque:
    if type(torq) == str:
      if 'N' in torq or 'n' in torq:  # выбираем изначения в измерениях Nm
        numb = '.'.join(re.findall('\d+', torq))  # выделяем из строки числовую часть
        if '.' in numb:  # обрабатываем дробные числа
          processed_torque.append(float(numb))
        else:  # обрабатываем целые числа
          processed_torque.append(float(int(numb)))
      elif 'k' in torq or 'K' in torq:  # выбираем значения в измерениях kgm
        numb = '.'.join(re.findall('\d+', torq))  # выделяем из строки числовую часть
        if '.' in numb:  # выбираем дробные числа
          numb = float(numb)
        else:  # выбираем целые числа
          numb = float(int(numb))
        numb = numb * 9.80665  # переводим kgm в Nm
        processed_torque.append(float(numb))
      else:  # обрабатываем значения без единиц измерения
        numb = '.'.join(re.findall('\d+', torq))  # выделяем из строки числовую часть
        processed_torque.append(float(int(numb)))
    else:
      processed_torque.append(torq)  # NaN будем оставоять нетронутыми

  return processed_torque

new_torque_train = fixing_torque(df_train['torque'])
df_train['torque'] = new_torque_train

new_torque_test = fixing_torque(df_test['torque'])
df_test['torque'] = new_torque_test


# Найдем медианы таблицы трейн и заполним пропуски сразу для двух таблиц

mileage_filler = df_train['mileage'].median()
df_train['mileage'] = df_train['mileage'].fillna(mileage_filler)
df_test['mileage'] = df_test['mileage'].fillna(mileage_filler)

engine_filler = df_train['engine'].median()
df_train['engine'] = df_train['engine'].fillna(engine_filler)
df_test['engine'] = df_test['engine'].fillna(engine_filler)

max_power_filler = df_train['max_power'].median()
df_train['max_power'] = df_train['max_power'].fillna(max_power_filler)
df_test['max_power'] = df_test['max_power'].fillna(max_power_filler)

torque_filler = df_train['torque'].median()
df_train['torque'] = df_train['torque'].fillna(torque_filler)
df_test['torque'] = df_test['torque'].fillna(torque_filler)

seats_filler = df_train['seats'].median()
df_train['seats'] = df_train['seats'].fillna(seats_filler)
df_test['seats'] = df_test['seats'].fillna(seats_filler)

max_torque_rpm_filler = df_train['max_torque_rpm'].median()
df_train['max_torque_rpm'] = df_train['max_torque_rpm'].fillna(max_torque_rpm_filler)
df_test['max_torque_rpm'] = df_test['max_torque_rpm'].fillna(max_torque_rpm_filler)


# Перевод трех столбцов к целым числам

df_train['engine'] = df_train['engine'].astype(int)
df_test['engine'] = df_test['engine'].astype(int)

df_train['seats'] = df_train['seats'].astype(int)
df_test['seats'] = df_test['seats'].astype(int)

df_train['max_torque_rpm'] = df_train['max_torque_rpm'].astype(int)
df_test['max_torque_rpm'] = df_test['max_torque_rpm'].astype(int)


#  Удаляем категориальные фичи

y_train = df_train['selling_price']
to_drop = ['selling_price', 'name', 'fuel', 'seller_type', 'transmission', 'owner']
X_train = df_train.drop(to_drop, axis=1)

y_test = df_test['selling_price']
to_drop = ['selling_price', 'name', 'fuel', 'seller_type', 'transmission', 'owner']
X_test = df_test.drop(to_drop, axis=1)


# Стандартизируем признаки

scaler = StandardScaler()

norm_x_train = scaler.fit_transform(X=X_train, y=y_train)
X_train_norm = pd.DataFrame(data=norm_x_train, columns = X_train.columns)

norm_x_test = scaler.fit_transform(X_test, y_test)
X_test_norm = pd.DataFrame(data=norm_x_test, columns = X_test.columns)


# Обучим Лассо-регрессию с ранее найденным альфа (из jupyter notebook)

model_lasso_with_alpha = Lasso(alpha=22810)
model_lasso_with_alpha.fit(X_train_norm, y_train)






