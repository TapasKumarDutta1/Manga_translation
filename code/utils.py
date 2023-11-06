import cv2


def put_text_in_box(
    img,
    text,
    box_coords,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1,
    font_color=(0, 0, 0),
    thickness=2,
):
    x, y, w, h = box_coords
    words = text.split()
    lines = []
    current_line = words[0]

    for word in words[1:]:
        test_line = current_line + " " + word
        (width, height), _ = cv2.getTextSize(test_line, font, font_scale, thickness)

        if width < w:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)

    y_offset = y + int((h - (len(lines) * (height + 5))) / 2)

    for line in lines:
        (width, height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.putText(img, line, (x, y_offset), font, font_scale, font_color, thickness)
        y_offset += height + 5

    return img
