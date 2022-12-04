from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from typing import List
import re
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import HW1_Regression as hw  # файл с выжимкой из основного jupyter notebook
from fastapi import  UploadFile
from fastapi.responses import FileResponse

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:

    # Веса для ранее обученной линейной регрессии:
    w1 = 40432.4358490244
    w2 = -0.8301428169347869
    w3 = -2314.458297336113
    w4 = 35.06125902081527
    w5 = 10256.952369795561
    w6 = 66.83388836032381
    w7 = -32809.48858710826
    w8 = -64.0559416940728
    w0 = -81381440.42943233

    year = item.year
    price = item.selling_price
    km_driven = item.km_driven
    mileage = item.mileage
    engine = item.engine
    max_power = item.max_power
    torque_and_rmp = item.torque
    seats = item.seats


    # Преобразуем mileage

    mil = mileage

    if type(mil) == str:  # способ обходить столбец, не трогая пустые значения
        mileage = float('.'.join(re.findall('\d+', mil)))  # выделяем из строки числовую часть
    else:
        mileage = 19.369999999999997  # пропуск заполняем медианой


    # Преобразуем engine

    eng = engine
    if type(eng) == str:  # способ обходить столбец, не трогая пустые значения
        engine = float('.'.join(re.findall('\d+', eng)))  # выделяем из строки числовую часть
    else:
        engine = 1248.0  # пропуск заполняем медианой


    # Преобразуем max_power

    power = max_power
    max_power_median = 81.86  # из раннее обработанных данных
    if type(power) == str:  # если значение не nan
        numb = '.'.join(re.findall('\d+', power))  # выделяем из строки числовую часть
        if numb == '':  # обрабатываем значения без чисел
            power = max_power_median
        elif '.' in numb:  # обрабатываем дробные числа
            power = float(numb)
        else:  # обрабатываем целые числа
            power = float(int(numb))
    else:  # если nan, то заменяем средним
        power = max_power_median

    max_power = power


    # Преобразуем torque

    torq = torque_and_rmp
    new_torque = []

    if type(torq) == str:  # способ обходить столбец, не трогая пустые значения
        torque = re.findall('\d+[.,\d+]*[ Nkg@mn]+', torq)  # находим все torque
        curr_torque = torque[0]
        if 'Nm' not in curr_torque and 'nm' not in curr_torque and 'kgm' not in curr_torque:  # если единицы измерения отсутствуют
            unit = re.findall('[NnKkGg]+[mM]', torq)  # находим единицы измерения
            if unit != []:  # если измерения нашлись
                curr_torque = torque[0] + unit[0]  # объединяем измерение и показатель
        new_torque.append(curr_torque)
    else:  # сохраняем пустые значения (nan)
        new_torque.append(160.0)

    for torq in new_torque:
        if type(torq) == str:
            if 'N' in torq or 'n' in torq:  # выбираем изначения в измерениях Nm
                numb = '.'.join(re.findall('\d+', torq))  # выделяем из строки числовую часть
                if '.' in numb:  # обрабатываем дробные числа
                    torque = float(numb)
                else:  # обрабатываем целые числа
                    torque = float(int(numb))
            elif 'k' in torq or 'K' in torq:  # выбираем значения в измерениях kgm
                numb = '.'.join(re.findall('\d+', torq))  # выделяем из строки числовую часть
                if '.' in numb:  # выбираем дробные числа
                    numb = float(numb)
                else:  # выбираем целые числа
                    numb = float(int(numb))
                numb = numb * 9.80665  # переводим kgm в Nm
                torque = float(numb)
            else:  # обрабатываем значения без единиц измерения
                numb = '.'.join(re.findall('\d+', torq))  # выделяем из строки числовую часть
                torque = float(int(numb))
        else:
            torque = 160.0    # преобразуем max_torque_rmp (получен из torque)


    # Выделим и преобразуем max_torque_rmp

    max_torque_rpm_median = 2400.0  # из раннее обработанных данных

    new_max_torque_rpm = []

    if type(torque_and_rmp) == str:  # если значение не nan
        max_torque = re.findall('[\d+,]{4}[\d+,. +\/-]*', torque_and_rmp)  # находим все max_torque_rpm
        if max_torque == []:  # если значение отсутствует, то заменяем средним
            new_max_torque_rpm.append(max_torque_rpm_median)
        else:  # сохраняем найденное значение
            curr_max_torque = max_torque[0]
            new_max_torque_rpm.append(curr_max_torque)
    else:  # если nan, то заменяем средним
        new_max_torque_rpm.append(max_torque_rpm_median)

    for torq in new_max_torque_rpm:
        if type(torq) == str:
            if '-' not in torq and '/' not in torq:  # обрабатываем обычные числа
                numb = int(''.join(re.findall('\d+', torq)))  # выделяем из строки числовую часть
            elif '/' in torq:  # обрабатываем диапозоны вида +/-
                numbs = re.findall('[\d,]+', torq)  # сохраняем все числа
                numb = int(''.join(re.findall('\d+', numbs[0])))  # первое число будет средним диапазона
            else:  # обрабатываем диапазоны
                # разделим диапазоны на два числа
                range_of_numbs = re.findall('[\d+,]+', torq)
                first_numb = int(''.join(re.findall('\d+', range_of_numbs[0])))
                second_numb = int(''.join(re.findall('\d+', range_of_numbs[1])))
                numb = int((first_numb + second_numb) / 2)  # сохраним среднее диапазона
            max_torque_rmp = numb
        else:  # если nan, то заменяем средним
            max_torque_rmp = max_torque_rpm_median

    # Подставим веса к обработанным признакам и предскажем стоимость автомобиля

    result = w0 + (w1 * year) + (w2 * km_driven) + (w3 * mileage) + (w4 * engine) + \
             (w5 * max_power) + (w6 * torque) + (w7 * seats) + (w8 * max_torque_rmp)

    # Тут я заметила что некоторые предсказания - отрицательные числа
    # Применим костыльное решение и будем брать модуль от предсказания...

    result = abs(result)

    return result



@app.post("/predict_items")
def predict_items(items: UploadFile):

    my_file = pd.read_csv(items.file)

    df_test = my_file.copy()

    # Обработаем необходимые нам столбцы, получив численные значения

    new_mileage_test = hw.fixing_mileage(df_test['mileage'])
    df_test['mileage'] = new_mileage_test

    new_engine_test = hw.fixing_engine(df_test['engine'])
    df_test['engine'] = new_engine_test

    new_max_power_test = hw.fixing_max_power(df_test['max_power'])
    df_test['max_power'] = new_max_power_test

    new_max_torque_rpm_test = hw.fixing_max_torque_rpm(df_test['torque'])
    df_test['max_torque_rpm'] = new_max_torque_rpm_test

    new_torque_test = hw.fixing_torque(df_test['torque'])
    df_test['torque'] = new_torque_test


    # Возьмем медианы из первоначальных данных и  заполним ими пропуски

    df_test['mileage'] = df_test['mileage'].fillna(hw.mileage_filler)

    df_test['engine'] = df_test['engine'].fillna(hw.engine_filler)

    df_test['max_power'] = df_test['max_power'].fillna(hw.max_power_filler)

    df_test['torque'] = df_test['torque'].fillna(hw.torque_filler)

    df_test['seats'] = df_test['seats'].fillna(hw.seats_filler)

    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].fillna(hw.max_torque_rpm_filler)

    #  Удаляем категориальные фичи

    y_test = df_test['selling_price']
    to_drop = ['selling_price', 'name', 'fuel', 'seller_type', 'transmission', 'owner']
    X_test = df_test.drop(to_drop, axis=1)

    # Стандартизируем признаки

    scaler = StandardScaler()
    norm_x_test = scaler.fit_transform(X_test, y_test)
    X_test_norm = pd.DataFrame(data=norm_x_test, columns=X_test.columns)

    # Полученные данные обработаны и готовы к предсказанию
    # Обучим модель лассо с ранее подобранной альфа на трейн данных из юпитер-ноутбука

    model_lasso_with_alpha = Lasso(alpha=22810)
    model_lasso_with_alpha.fit(hw.X_train_norm, hw.y_train)

    # Предскажем цену для полученного файла

    pred_lasso = model_lasso_with_alpha.predict(X_test_norm)

    # Применим все то же костыльное решение - брать модуль от каждого предсказания...

    for i in range(len(pred_lasso)):
        pred_lasso[i] = abs(pred_lasso[i])

    # Объединим предсказания и первоначальную таблицу

    df_item = my_file.copy()

    df_item['predicted_price'] = pred_lasso

    # Сохраним файл на компьютере

    path = "/home/twilyfm/PycharmProjects/pythonProject/result_csv_file.csv"
    df_item.to_csv(path)

    return FileResponse(path)

