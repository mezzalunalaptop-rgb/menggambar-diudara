import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Otomatis download model AI Hand Landmarker jika belum ada
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
  print('Sedang mendownload model AI Hand Landmarker...')
  urllib.request.urlretrieve(
      'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      model_path,
  )

# Inisialisasi MediaPipe Tasks Hand Landmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

# Buka Kamera dengan resolusi optimal
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
  print('ERROR: Kamera tidak dapat dibuka!')
  exit()

paintWindow = None
prev_x, prev_y = 0, 0
smooth_x, smooth_y = 0, 0

# Faktor smoothing (0.7 = responsif cepat dan mengalir mulus tanpa patah-patah)
ALPHA = 0.7

lost_frame_count = 0
MAX_LOST_FRAMES = 8  # Toleransi agar garis tidak mudah terputus

# Warna kuas (Format BGR: Biru, tebal garis 8)
brushColor = (255, 0, 0)
brushThickness = 8

print('Sistem Virtual Air Writing (Omni-Directional Pointing) Berjalan.')
print('Miringkan tangan atau tunjuk ke depan bebas. Genggam untuk berhenti.')
print('Tekan "c" untuk membersihkan layar, "q" untuk keluar.')

while cap.isOpened():
  success, frame = cap.read()
  if not success or frame is None:
    print('Gagal membaca frame dari kamera.')
    break

  # Balik gambar seperti cermin agar natural
  frame = cv2.flip(frame, 1)
  h, w, _ = frame.shape

  if paintWindow is None:
    paintWindow = np.zeros((h, w, 3), dtype=np.uint8)

  # Konversi frame ke MediaPipe Image
  mp_image = mp.Image(
      image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  )
  detection_result = detector.detect(mp_image)

  is_drawing_gesture = False
  center = None

  if detection_result.hand_landmarks:
    for hand_landmarks in detection_result.hand_landmarks:
      # Titik penting: Pergelangan (0), Ujung Telunjuk (8), Ruas (7), PIP (6)
      wrist = hand_landmarks[0]
      index_tip = hand_landmarks[8]
      index_dip = hand_landmarks[7]
      index_pip = hand_landmarks[6]

      # Jari tengah untuk deteksi genggaman (12 & 10)
      middle_tip = hand_landmarks[12]
      middle_pip = hand_landmarks[10]

      # Deteksi gestur menunjuk yang tahan terhadap berbagai sudut kemiringan tangan
      dist_tip_wrist = np.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y)
      dist_pip_wrist = np.hypot(index_pip.x - wrist.x, index_pip.y - wrist.y)

      is_index_extended = dist_tip_wrist > (dist_pip_wrist * 1.1)
      is_middle_folded = middle_tip.y > middle_pip.y

      if is_index_extended and is_middle_folded:
        is_drawing_gesture = True

        # EKTRAPOLASI ADAPTIF: Menyesuaikan apakah tangan miring atau menunjuk lurus ke depan
        # Menghitung panjang vektor ruas telunjuk untuk menggeser titik pas di ujung kuku
        dx = index_tip.x - index_dip.x
        dy = index_tip.y - index_dip.y
        vector_len = np.hypot(dx, dy)

        if vector_len > 0.001:
          # Normalisasi dan beri dorongan pas di ujung kuku
          offset_scale = 0.35
          tip_x = index_tip.x + (dx / vector_len) * (vector_len * offset_scale)
          tip_y = index_tip.y + (dy / vector_len) * (vector_len * offset_scale)
        else:
          tip_x, tip_y = index_tip.x, index_tip.y

        cx, cy = int(tip_x * w), int(tip_y * h)
        center = (cx, cy)

      # Gambar kerangka tangan tipis sebagai panduan visual
      for lm in hand_landmarks:
        lx, ly = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (lx, ly), 2, (0, 255, 0), cv2.FILLED)

  # Logika Menggambar dengan Exponential Smoothing Tingkat Lanjut
  if is_drawing_gesture and center is not None:
    lost_frame_count = 0

    if smooth_x == 0 and smooth_y == 0:
      smooth_x, smooth_y = center[0], center[1]
    else:
      smooth_x = int(ALPHA * center[0] + (1 - ALPHA) * smooth_x)
      smooth_y = int(ALPHA * center[1] + (1 - ALPHA) * smooth_y)

    current_point = (smooth_x, smooth_y)

    # Lingkaran kuning penanda tepat di ujung kuku
    cv2.circle(frame, current_point, 9, (0, 255, 255), cv2.FILLED)

    if prev_x == 0 and prev_y == 0:
      prev_x, prev_y = current_point

    # Gambar garis mulus dengan Anti-Aliasing (LINE_AA) agar tidak patah-patah
    cv2.line(
        paintWindow,
        (prev_x, prev_y),
        current_point,
        brushColor,
        brushThickness,
        cv2.LINE_AA,
    )
    prev_x, prev_y = current_point
  else:
    if prev_x != 0 and prev_y != 0:
      lost_frame_count += 1
      if lost_frame_count > MAX_LOST_FRAMES:
        prev_x, prev_y = 0, 0
        smooth_x, smooth_y = 0, 0
    else:
      prev_x, prev_y = 0, 0
      smooth_x, smooth_y = 0, 0

  # Gabungkan kanvas lukis dengan frame kamera secara mulus
  gray = cv2.cvtColor(paintWindow, cv2.COLOR_BGR2GRAY)
  _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
  frame = cv2.bitwise_and(frame, frame, mask=inv)
  frame = cv2.bitwise_or(frame, paintWindow)

  # Tampilkan status di layar
  status_text = (
      'Mode: MENGGAMBAR (Aktif)'
      if is_drawing_gesture
      else 'Mode: STANDBY / GENGGAM'
  )
  status_color = (0, 255, 0) if is_drawing_gesture else (0, 0, 255)

  cv2.putText(
      frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2
  )
  cv2.putText(
      frame,
      "Tekan 'c': Hapus | 'q': Keluar",
      (10, 60),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (255, 255, 255),
      1,
  )

  # Tampilkan jendela program
  cv2.imshow('Virtual Air Writing - Omni Directional', frame)

  # Kontrol Keyboard
  key = cv2.waitKey(1) & 0xFF
  if key == ord('c'):
    paintWindow = np.zeros((h, w, 3), dtype=np.uint8)
  elif key == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()