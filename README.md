# Дз_1 ML

### Что было сделано:
* Обработаны первоначальные данные:
  * числовые признаки отредактированы
  * категориальные приведены к числовым значениям
  * произведена стандартизация данных
  * обработаны пропуски
* Была попытка создать новые признаки на основе уже имеющихся, а также попытка собрать данные о классе автомобиля
* Было обученно несколько моделей линейной регрессии (обычная, с Lasso-, Ridge- и ElasticNet-регуляризацией) на данных разного этапа обработки
* Организован сервис на FastApi, предсказывающий цену для одного автомобиля и группы автомобилей 

### Результаты:
* Качество предсказания моделей довольно среднее: R2 для лучшей модели составили 0.659 для трейна и 0.599 для теста. Это показатели обычной линейной регрессии для данных с добавлением категориальных признаков. Однако почему-то показатели business_metrics для этих данных оказались хуже, чем для данных без категориальных признаков
* Результаты организованного сервиса на FastApi

### Возникшие проблемы:
* Не удалось обработать выбросы так как я испугалась потерять данные о дорогих, спортивных или мощных автомобилях
* Модели предсказывают не очень хорошо, некоторые предсказания цен были отрицательными...
* Хотелось бы более качественно провести Feature Engineering
