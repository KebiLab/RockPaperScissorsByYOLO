import sys
import os
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5 import uic
from ultralytics import YOLO


class FolderProcessor(QThread):
    finished = pyqtSignal(dict)

    def __init__(self, folder_path, model, conf_thresh=0.25, iou_thresh=0.5):
        super().__init__()
        self.folder = folder_path
        self.yolo_model = model
        self.conf_t = conf_thresh
        self.iou_t = iou_thresh

    def calc_iou(self, b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1, a2 = (b1[2] - b1[0]) * (b1[3] - b1[1]), (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter) if inter > 0 else 0

    def parse_gt(self, lbl_path):
        boxes = []
        if not os.path.exists(lbl_path):
            return boxes
        with open(lbl_path, 'r') as f:
            for ln in f:
                p = ln.strip().split()
                if len(p) >= 5:
                    cls = int(p[0])
                    if cls == 0:
                        xc, yc, w, h = map(float, p[1:5])
                        boxes.append([xc - w/2, yc - h/2, xc + w/2, yc + h/2])
        return boxes

    def run(self):
        img_dir, lbl_dir = os.path.join(self.folder, 'images'), os.path.join(self.folder, 'labels')
        ext = ('.jpg', '.jpeg', '.png', '.bmp')
        imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(ext)]
        total = len(imgs)
        detections, gts, confs_list = [], [], []

        for fname in imgs:
            frm = cv2.imread(os.path.join(img_dir, fname))
            if frm is None:
                continue
            h, w = frm.shape[:2]
            gt_boxes = self.parse_gt(os.path.join(lbl_dir, fname.rsplit('.', 1)[0] + '.txt'))
            gt_abs = [[b[0]*w, b[1]*h, b[2]*w, b[3]*h] for b in gt_boxes]
            res = self.yolo_model(frm, verbose=False, conf=self.conf_t)
            det_b, det_c = [], []
            for r in res:
                if r.boxes is not None:
                    for b in r.boxes:
                        if int(b.cls[0]) == 0:
                            det_b.append(list(map(float, b.xyxy[0])))
                            cf = float(b.conf[0])
                            det_c.append(cf)
                            confs_list.append(cf)
            gts.append({'boxes': gt_abs, 'matched': [False]*len(gt_abs)})
            detections.append({'boxes': det_b, 'confs': det_c})

        tp, fp = 0, 0
        for dd, gd in zip(detections, gts):
            gm = gd['matched']
            for det_box, det_cf in zip(dd['boxes'], dd['confs']):
                best_iou, best_idx = 0, -1
                for j, gt_b in enumerate(gd['boxes']):
                    if not gm[j]:
                        iou_v = self.calc_iou(det_box, gt_b)
                        if iou_v > best_iou:
                            best_iou, best_idx = iou_v, j
                if best_iou >= self.iou_t and best_idx >= 0:
                    tp += 1
                    gm[best_idx] = True
                else:
                    fp += 1

        fn = sum(1 for g in gts for m in g['matched'] if not m)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        self.finished.emit({
            'total': total, 'tp': tp, 'fp': fp, 'fn': fn,
            'map50': 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        })


class RockPaperScissorsApp(QMainWindow):
    GEST = {0: 'Paper', 1: 'Rock', 2: 'Scissors'}

    def __init__(self):
        super().__init__()
        uic.loadUi('main.ui', self)
        self.cam_cb.clear()
        self.cam_cb.addItem("Вебкамера (0)", 0)
        self.btn_start_turnir.clicked.connect(self.start_turnir)
        self.btn_stop_turnir.clicked.connect(self.stop_turnir)
        self.btn_start_detect.clicked.connect(self.start_detect)
        self.btn_stop_detect.clicked.connect(self.stop_detect)
        self.btn_test_folder.clicked.connect(self.test_folder)
        self.model = YOLO('dlyaturnira.pt')
        self.cap = None
        self.is_running = False
        self.mode = None
        self.fps_val, self.fps_c = 0, 0
        self.fps_t = cv2.getTickCount()
        self.turnir_active = False
        self.gestures_captured = False
        self.timer = None
        self.proc = None
        self.wait_label = None
        self.left_best = None
        self.right_best = None

    def start_turnir(self):
        if self.is_running and self.mode == 'turnir':
            return
        self.stop_all()
        self.cap = cv2.VideoCapture(self.cam_cb.currentData())
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть камеру {self.cam_cb.currentData()}")
            return
        self.is_running = True
        self.mode = 'turnir'
        self.btn_start_turnir.setEnabled(False)
        self.btn_stop_turnir.setEnabled(True)
        self.btn_start_detect.setEnabled(False)
        self.btn_stop_detect.setEnabled(False)
        self.btn_test_folder.setEnabled(False)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        self.turnir_active = True
        self.gestures_captured = False
        self.status_label.setText("Статус: Покажите жесты!")

    def stop_turnir(self):
        if self.mode != 'turnir':
            return
        self.turnir_active = False
        self.gestures_captured = False
        self.mode = None
        self.is_running = False
        self.btn_start_turnir.setEnabled(True)
        self.btn_stop_turnir.setEnabled(False)
        self.btn_start_detect.setEnabled(True)
        self.btn_test_folder.setEnabled(True)
        if self.timer and self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.clear()
        self.status_label.setText("Статус: Ожидание")

    def start_detect(self):
        if self.is_running and self.mode == 'detect':
            return
        self.stop_all()
        self.cap = cv2.VideoCapture(self.cam_cb.currentData())
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть камеру {self.cam_cb.currentData()}")
            return
        self.is_running = True
        self.mode = 'detect'
        self.btn_start_detect.setEnabled(False)
        self.btn_stop_detect.setEnabled(True)
        self.btn_start_turnir.setEnabled(False)
        self.btn_stop_turnir.setEnabled(False)
        self.btn_test_folder.setEnabled(False)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        self.status_label.setText("Статус: Детекция...")

    def stop_detect(self):
        if self.mode != 'detect':
            return
        self.mode = None
        self.is_running = False
        self.btn_start_detect.setEnabled(True)
        self.btn_stop_detect.setEnabled(False)
        self.btn_start_turnir.setEnabled(True)
        self.btn_test_folder.setEnabled(True)
        if self.timer and self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.clear()
        self.status_label.setText("Статус: Ожидание")

    def stop_all(self):
        self.turnir_active = False
        self.gestures_captured = False
        self.left_best = None
        self.right_best = None
        if self.timer and self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.clear()

    def determine_winner(self, left, right):
        if left == right:
            return "НИЧЬЯ"
        if (left == 0 and right == 2) or (left == 1 and right == 0) or (left == 2 and right == 1):
            return "Победил игрок СПРАВА"
        return "Победил игрок СЛЕВА"

    def capture_gests(self):
        if self.gestures_captured:
            return
        if self.left_best and self.right_best:
            self.gestures_captured = True
            winner = self.determine_winner(self.left_best[0], self.right_best[0])
            left_name = self.GEST.get(self.left_best[0], 'Unknown')
            right_name = self.GEST.get(self.right_best[0], 'Unknown')
            msg = f"{winner}\n\nСЛЕВА: {left_name}\nСПРАВА: {right_name}"
            QMessageBox.information(self, "Результат", msg)
            self.status_label.setText(f"Статус: {winner}")
            self.turnir_active = False
        elif self.left_best or self.right_best:
            self.gestures_captured = True
            detected = "СЛЕВА" if self.left_best else "СПРАВА"
            self.status_label.setText(f"Статус: Обнаружен только игрок {detected}")
            self.turnir_active = False
        else:
            self.gestures_captured = True
            self.status_label.setText("Статус: Жесты не обнаружены")
            self.turnir_active = False

    def show_frame(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        pm = QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888))
        self.video_label.setPixmap(pm.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def show_wait(self):
        self.wait_label = QLabel("Обработка...", self)
        self.wait_label.setStyleSheet("background-color: rgba(0, 0, 0, 150); color: white; font-size: 16px; padding: 8px; border-radius: 5px;")
        self.wait_label.setAlignment(Qt.AlignCenter)
        self.wait_label.setGeometry((self.width() - 300) // 2, (self.height() - 40) // 2, 300, 40)
        self.wait_label.show()

    def hide_wait(self):
        if self.wait_label:
            self.wait_label.hide()
            self.wait_label = None
        self.btn_test_folder.setEnabled(True)

    def on_finished(self, res):
        self.hide_wait()
        QMessageBox.information(self, "Тест", f"Готово!\n\nmAP50: {res['map50']:.1%}")

    def update_frame(self):
        if not self.cap or not self.cap.isOpened() or not self.is_running:
            return
        ok, frm = self.cap.read()
        if not ok:
            return
        h, w = frm.shape[:2]
        mid_x = w // 2
        res = self.model(frm, verbose=False, conf=0.5)
        out = frm.copy()
        cv2.line(out, (mid_x, 0), (mid_x, h), (0, 255, 0), 2)
        self.left_best, self.right_best = None, None
        for r in res:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls, cf = int(b.cls[0]), float(b.conf[0])
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cx = (x1 + x2) / 2
                cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
                gest_name = self.GEST.get(cls, 'Unknown')
                lbl = f'{gest_name} {cf*100:.0f}%'
                cv2.putText(out, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                if cx < mid_x:
                    if self.left_best is None or cf > self.left_best[1]:
                        self.left_best = (cls, cf, (x1, y1, x2, y2))
                else:
                    if self.right_best is None or cf > self.right_best[1]:
                        self.right_best = (cls, cf, (x1, y1, x2, y2))
        if self.mode == 'turnir' and not self.gestures_captured:
            if self.left_best and self.right_best:
                self.capture_gests()
        self.fps_c += 1
        t = cv2.getTickCount()
        elapsed = (t - self.fps_t) / cv2.getTickFrequency()
        if elapsed >= 1.0:
            self.fps_val = self.fps_c / elapsed
            self.fps_c, self.fps_t = 0, t
        cv2.putText(out, f'FPS: {self.fps_val:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self.show_frame(out)

    def test_folder(self):
        self.model = YOLO('kebilab.pt')
        folder = QFileDialog.getExistingDirectory(self, "Папка с тестовыми данными")
        if not folder:
            return
        img_p, lbl_p = os.path.join(folder, 'images'), os.path.join(folder, 'labels')
        if not os.path.exists(img_p) or not os.path.exists(lbl_p):
            QMessageBox.warning(self, "Ошибка", "Нужны подпапки 'images' и 'labels'")
            return
        self.show_wait()
        self.proc = FolderProcessor(folder, self.model)
        self.proc.finished.connect(self.on_finished)
        self.proc.start()

    def closeEvent(self, ev):
        self.stop_all()
        if self.proc and self.proc.isRunning():
            self.proc.terminate()
        ev.accept()


def main():
    app = QApplication(sys.argv)
    win = RockPaperScissorsApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
