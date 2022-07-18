from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import sys
from unicodedata import category
chrs = (chr(i) for i in range(sys.maxunicode + 1))
punctuation = set(c for c in chrs if category(c).startswith("P")) -\
              {'\'', '\"', '(', '“', '‘', '《', '[', '{', '<'}

def get_text_width(ttfont, font_size, text):
    return ttfont.getsize(text)[0] / font_size

def split_text(ttfont, font_size, text, col_width):
    if len(text) == 1:
        return [text]
    text = text.strip('\n')
    text_slice = []
    slice_start, text_width, text_length = 0, 0, 0
    for i, ch in enumerate(text):
        if ch == '\n':
            if text_width == 0:
                slice_start += 1
            else:
                slice_end = slice_start + text_length
                text_slice.append(text[slice_start: slice_end])
                slice_start = slice_end + 1
                text_width, text_length = 0, 0
            continue
        ch_width = get_text_width(ttfont, font_size, ch)
        text_width += ch_width
        to_split = False
        if text_width > col_width:
            if ch not in punctuation:
                to_split = True
            else:
                if len(text) > i + 1 and text[i + 1] in punctuation:
                    to_split = True

        if to_split:
            slice_end = slice_start + text_length
            text_slice.append(text[slice_start: slice_end])
            slice_start = slice_end
            text_width, text_length = ch_width, 1
        else:
            text_length += 1
    if slice_start < len(text):
        text_slice.append(text[slice_start:])

    return text_slice


def add_text_in_box(box, add_text, image, font_path, border_width=0, **kwargs):
    if not add_text:
        return
    (x1, y1), (x2, y2) = box
    box_width = x2 - x1
    box_height = y2 - y1

    aspect_ratio = box_width / box_height
    box_area_ratio = box_width * box_height / image.size[0] / image.size[1]

    bias = 0.15 * (1 if aspect_ratio < 1 else 1 / aspect_ratio) *\
           min(12 * box_area_ratio, 2) +\
           0.2 * (1 / max(1, len(add_text) - 2))

    text_area_ratio_max = 0.92 - bias
    text_area_ratio_min = 0.8 - bias * 1.4

    font_size = int((box_width + box_height) // 30) + 1
    last_adjust_type = 0  # 1 increase, -1 decrease

    # calculate font size
    while True:
        max_row_num, row_remain = divmod(box_height, font_size)
        max_col_num, col_remain = divmod(box_width, font_size)
        max_row_num = int(max_row_num)
        max_col_num = int(max_col_num)
        col_border = 1 if max_col_num > 1 else 0
        row_border = 1 if max_row_num > 1 else 0
        max_text_num = (max_row_num - row_border) * (max_col_num - col_border)

        ttfont = ImageFont.truetype(font_path, font_size)
        text_real_length = get_text_width(ttfont, font_size, add_text)
        text_area_ratio = text_real_length / max_text_num

        if text_area_ratio > text_area_ratio_max:
            font_size -= 1
            if last_adjust_type == 1:
                break
            last_adjust_type = -1

        elif text_area_ratio < text_area_ratio_min:
            if last_adjust_type == -1:
                break
            if font_size == min(int(box_height), int(box_width)):
                break
            font_size += 1
            last_adjust_type = 1
        else:
            break

    # add text
    col_num = max_col_num - col_border
    text_slice = split_text(ttfont, font_size, add_text, col_num)
    need_row = len(text_slice)
    start_row = (max_row_num - need_row) / 2

    draw = ImageDraw.Draw(image)
    if border_width:
        border_color = kwargs.get('border_color', 'red')
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=None, outline=border_color, width=border_width)

    mask = Image.new('RGB', image.size, 'black')
    mask_draw = ImageDraw.Draw(mask)

    every_text_width = [get_text_width(ttfont, font_size, (text[:-1] if text[-1] in punctuation else text))
                        for text in text_slice]
    max_text_width = max(every_text_width)
    start_col = (max_col_num - max_text_width) / 2
    for i, text in enumerate(text_slice):
        x = x1 + start_col * font_size + col_remain / 2
        y = y1 + (start_row + i) * font_size + row_remain / 2
        mask_draw.text((x, y), text, fill='white', font=ttfont)

    binary_mask = cv2.cvtColor(np.asarray(mask), cv2.COLOR_RGB2GRAY)
    font_alpha = binary_mask.copy()
    kernel_size = font_size // 10 + 1
    dilate_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilate_mask = cv2.dilate(binary_mask, dilate_kernel, iterations=1)
    outer_alpha = dilate_mask.copy()

    font_color = kwargs.get('font_color', 'black')
    outer_image = Image.new('RGBA', image.size, 'white')
    image.paste(outer_image, Image.fromarray(outer_alpha))
    font_color_image = Image.new('RGBA', image.size, font_color)
    image.paste(font_color_image, Image.fromarray(font_alpha))


if __name__ == '__main__':
    # test
    add_text = '测试样例。，？abcd+-*/' * 3
    oriImg = Image.new('RGB', (600, 250), 'black')
    points = [(100, 100), (oriImg.size[0] - 100, oriImg.size[1])]
    font_path = "font/宋体-粗体.ttf"
    add_text_in_box(points, add_text, oriImg, font_path, border_width=2)
    oriImg.show()
