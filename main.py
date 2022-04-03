import cv2
import numpy as np
import math

avatar = (cv2.imread('avatars/avatar0.jpeg'), cv2.imread('avatars/avatar1.jpeg'), cv2.imread('avatars/avatar2.jpeg'),
          cv2.imread('avatars/avatar3.jpeg'), cv2.imread('avatars/avatar4.jpeg'), cv2.imread('avatars/avatar5.jpeg'))
color = ((218, 32, 42), (170, 0, 211), (116, 0, 255), (74, 113, 74), (64, 187, 255), (113, 248, 249))

locations = np.ones((2000, 2000, 3), dtype=np.int8)  # Карта для отслеживания найденных лиц в последних 5 кадрах
# avatar_id, how_many_frames_ago, color_id для каждой координаты

events = []  # Массив точек интереса
faces = cv2.CascadeClassifier('faces.xml')
profiles = cv2.CascadeClassifier('profiles.xml')


# Функция обработки кадра каскадом
def search(cscd, img_s):
    img_gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
    result = cscd.detectMultiScale(img_gray, minSize=[40, 40], scaleFactor=1.3, minNeighbors=5)
    return result


vid = cv2.VideoCapture(0)  # Live WebCam

# Инициализация карты
for i in range(2000):
    for j in range(2000):
        locations[i, j, 1] = 10

cnt = 1  # Новое лицо номер cnt
while True:
    success, img = vid.read()  # Текущий кадр
    img_h, img_w = img.shape[:2]  # Высота и ширина текущего кадра

    # Запускаем поиск лица в анфас и лица в профиль и добавляем к найденным лицам
    found_faces = []
    found_faces.extend(search(faces, img))
    found_faces.extend(search(profiles, img))
    # Запускаем поиск в профиль на отзеркаленной по оси Y картинке
    mirror = search(profiles, cv2.flip(img, 1))
    # Зеркалим полученные координаты в исходную ориентацию и добавляем к найденным лицам
    for i in range(len(mirror)):
        mirror[i, 0] = img_w - mirror[i, 0] - mirror[i, 2]
    found_faces.extend(mirror)

    # Проходим по всем найденным лицам
    for (x, y, w, h) in found_faces:
        old_flag = False  # Флаг, проверяющий было ли лицо рядом с текущими координатами в прошлом кадре
        redundant_flag = False  # Флаг, проверяющий было ли найдено одно лицо несколькими каскадами в текущем кадре
        avatar_id = 1
        color_id = 1

        # Проверка в радиусе pxls1 вокруг текущего лица на уже найденность
        pxls1 = 60
        for (check_x, check_y) in events:
            if (math.sqrt((check_x - x) ** 2 + (check_y - y) ** 2) < 60) and (locations[check_y, check_x, 1] == 0):
                redundant_flag = True

        # Если это лицо еще не было найдено
        if not redundant_flag:
            # Добавляем координаты в массив точек интереса
            events.append((x, y))

            # Проверка в квадрате с центром в (x, y) и стороной 2*pxls2 на бывшие там лица за последние 5 кадров
            pxls2 = 50
            for i in range(max((y - pxls2), 0), min((y + pxls2), img_h)):
                for j in range(max((x - pxls2), 0), min((x + pxls2), img_w)):
                    # Если недавно рядом уже было лицо, считаем их одним человеком
                    if locations[i, j, 1] < 5:
                        old_flag = True
                        avatar_id = locations[i, j, 0]
                        color_id = locations[i, j, 2]
                        break
                if old_flag:
                    break

            # Если лиц в квадрате недавно не было, считаем новым человеком
            if not old_flag:
                avatar_id = cnt
                color_id = cnt
                cnt += 1

            # Обновляем карту
            locations[y, x, 0] = avatar_id
            locations[y, x, 1] = 0
            locations[y, x, 2] = color_id

            # Вставляем аватар
            avatar_cur = avatar[abs((avatar_id - 1) % 6)]
            cv2.rectangle(img, (x, y), (x + w, y + h), color[min(abs(color_id // 6), 5)], thickness=15)
            avatar_insert = cv2.resize(avatar_cur, (w, h))
            img[y:(y + h), x:(x + w)] = avatar_insert

    cv2.imshow("Furryfication", img)  # Вывод обработанного кадра

    # Обновляем массив точек интереса на +1 кадр по давности и удаляем точки неактивные больше 5 кадров
    i = 0
    while i < len(events):
        x, y = events[i]
        if locations[y, x, 1] < 5:
            locations[y, x, 1] += 1
            i += 1
        else:
            del events[i]

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

