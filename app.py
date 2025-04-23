from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

import cv2
import mediapipe as mp
import numpy as np
# High level API mediapipe kurulumu
def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils


    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # eldeki cizgiler
                
                lm_list = []

                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy))

                    if len(lm_list) >= 13:
                        x1, y1 = lm_list[4]  # Bas parmak
                        x2, y2 = lm_list[8]  # isaret parmagi
                        x3, y3 = lm_list[12] # orta parmak

                        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
                        cv2.circle(img, (x3, y3), 10, (255, 0, 0), cv2.FILLED)

                        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

                        length = np.hypot(x2 - x1, y2 - y1)
                        lenc = np.hypot(x3 - x2, y3 - y2)
                        vol_bar = np.interp(length, [30, 200], [400, 150])
                        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)
                        cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
                        #max 200 min 30 30'a sifir dersek 100'ede 200 aralik ozaman 170 olur 100/170'den her uzunlugu 10/17 ile carpip sesi ayarlayacak
                        ses =length/250
                        if lenc > 50:
                            if (length/250)>0 and (length/250) < 1:
                                volume.SetMasterVolumeLevelScalar(ses, None)
                                print(lenc)
                        else:
                            volume.SetMasterVolumeLevelScalar(0.5, None)
                            break
                        
                        #volume.SetMasterVolumeLevelScalar(((length-30)/(10/17))*0.01, None) #51%
                    

                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)


        cv2.imshow('Hand Tracking', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()