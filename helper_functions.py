import cv2
import numpy as np

def resize_video(width, height, max_width = 600):

  if (width > max_width):
  
    proportion = width / height
    
    video_width = max_width
    video_height = int(video_width / proportion)
  else:
    video_width = width
    video_height = height

  return video_width, video_height