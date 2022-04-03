import cv2
import numpy as np
import math

avatar = (cv2.imread('avatars/avatar0.jpeg'), cv2.imread('avatars/avatar1.jpeg'), cv2.imread('avatars/avatar2.jpeg'),
          cv2.imread('avatars/avatar3.jpeg'), cv2.imread('avatars/avatar4.jpeg'), cv2.imread('avatars/avatar5.jpeg'))
color = ((255, 0, 0), (0, 0, 255), (0, 255, 0))
locations = np.ones((2000, 2000, 2), dtype=np.int8)  # Карта для отслеживания найденных лиц в последних 10 кадрах
events = []
faces = cv2.CascadeClassifier('faces.xml')
profiles = cv2.CascadeClassifier('profiles.xml')


def search(cscd, img_s):
    img_gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
    result = cscd.detectMultiScale(img_gray, minSize=[40, 40], scaleFactor=1.2, minNeighbors=5)
    return result


#vid = cv2.VideoCapture('videos/sample1.mp4')
vid = cv2.VideoCapture(0)  # Live WebCam

for i in range(2000):
    for j in range(2000):
        locations[i, j, 1] = 10

cnt = 1  # Свободный аватар

while True:
    success, img = vid.read()
    #img = cv2.resize(img, (1280, 720))
    img_h, img_w = img.shape[:2]

    # Запускаем поиск лица прямо и лица в профиль и добавляем к найденным лицам
    found_faces = []
    found_faces.extend(search(faces, img))
    found_faces.extend(search(profiles, img))
    # Запускаем поиск в профиль на отзеркаленной по оси Y картинке
    mirror = search(profiles, cv2.flip(img, 1))
    # Зеркалим полученные координаты в исходную орентацию и добавляем к найденным лицам
    for i in range(len(mirror)):
        mirror[i, 0] = img_w - mirror[i, 0] - mirror[i, 2]
    found_faces.extend(mirror)
    # Сортируем найденные лица в порядке возрастания для перекрытия при выводе
    found_faces.sort(key=lambda size: size[3])

    for (x, y, w, h) in found_faces:
        flag = False
        avatar_id = 1
        events.append((x, y))
        pxls = 50
        for i in range(max((y - pxls), 0), min((y + pxls), img_h)):
            for j in range(max((x - pxls), 0), min((x + pxls), img_w)):
                if locations[i, j, 1] < 5:
                    flag = True
                    avatar_id = locations[i, j, 0]
                    break
            if flag:
                break

        if not flag:
            avatar_id = cnt
            cnt += 1
        #print('avatar_id = ', avatar_id, 'cnt = ', cnt)
        locations[y, x, 0] = avatar_id
        locations[y, x, 1] = 0

        avatar_cur = avatar[(avatar_id - 1) % 6]
        cv2.rectangle(img, (x, y), (x + w, y + h), color[min((cnt - 1) // 6, 2)], thickness=3)
        avatar_insert = cv2.resize(avatar_cur, (w, h))
        img[y:(y + h), x:(x + w)] = avatar_insert

    cv2.imshow("Anon", img)

    for_del = []

    i = 0
    while i < len(events):
        x, y = events[i]
        if locations[y, x, 1] < 5:
            locations[y, x, 1] += 1
            i += 1
        else:
            del events[i]

    # for i in range(len(for_del)):
    #     del events[for_del[i]]

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

