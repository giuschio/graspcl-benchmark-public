import cv2
import numpy as np
import pyrealsense2 as rs


WIDTH = 640
HEIGHT = 480
FPS = 30
WINDOW_NAME = "RealSense IR + Depth"


def colorize_depth(depth_frame):
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
    return cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)


def prepare_ir(ir_frame):
    ir_image = np.asanyarray(ir_frame.get_data())
    return cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS)

    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            ir_frame = frames.get_infrared_frame(1)

            if not depth_frame or not ir_frame:
                continue

            ir_view = prepare_ir(ir_frame)
            depth_view = colorize_depth(depth_frame)
            combined_view = np.hstack((ir_view, depth_view))

            cv2.putText(
                combined_view,
                "Infrared",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                combined_view,
                "Depth",
                (WIDTH + 12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, combined_view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
