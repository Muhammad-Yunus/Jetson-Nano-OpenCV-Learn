import cv2
import numpy as np

# draw_ped() function to draw bounding box with top labeled text
def draw_ped(img, label, x0, y0, xt, yt, font_size=0.4, alpha=0.5, bg_color=(255,0,0), ouline_color=(255,255,255), text_color=(0,0,0)):
    overlay = np.zeros_like(img)

    y0, yt = max(y0 - 15, 0) , min(yt + 15, img.shape[0])
    x0, xt = max(x0 - 15, 0) , min(xt + 15, img.shape[1])

    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
    cv2.rectangle(overlay,
                    (x0, y0 + baseline),  
                    (max(xt, x0 + w), yt), 
                    bg_color, 
                    -1)
    cv2.rectangle(img,
                    (x0, y0 + baseline),  
                    (max(xt, x0 + w), yt), 
                    ouline_color, 
                    2)
    pts = np.array([[x0, y0 - h - baseline], # top left
                    [x0 + w, y0 - h - baseline], # top right
                    [x0 + w + 10, y0 + baseline], # bolom right
                    [x0,y0 + baseline]]) # bottom left
    cv2.fillPoly(img, [pts], ouline_color) # add label white fill
    cv2.polylines(img, [pts], True, ouline_color, 2) # add label white border 
    cv2.putText(img, 
                label, 
                (x0, y0),                   
                cv2.FONT_HERSHEY_SIMPLEX,     
                font_size,                          
                text_color,                
                1,
                cv2.LINE_AA) 

    img_blend = cv2.addWeighted(img, 1, overlay, alpha, 0.0)

    return img_blend