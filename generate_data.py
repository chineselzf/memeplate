import os
import json
from PIL import Image
from add_text_utils import add_text_in_box
from tqdm import tqdm
from multiprocessing import Pool, Queue

DATA_ROOT_DIR = 'dataset'
NUM_WORKER = 12
FONT_PATH = "font/宋体-粗体.ttf"

if not os.path.exists(os.path.join(DATA_ROOT_DIR, 'full_images')):
    os.mkdir(os.path.join(DATA_ROOT_DIR, 'full_images'))

task_queue = Queue()


def read_meta_data():
    meme_info_list = []
    for split_part in ['train', 'dev', 'test']:
        with open(os.path.join(DATA_ROOT_DIR, '{}_data.json'.format(split_part)), 'r', encoding='utf-8') as f:
            meme_info_list += json.load(f)
    with open(os.path.join(DATA_ROOT_DIR, 'source_image_info.json'), 'r', encoding='utf-8') as f:
        template_info_list = json.load(f)
        template_info_map = {x['source_image_id']:x for x in template_info_list}

    meme_info_map = {}
    for x in meme_info_list:
        meme_info_map.setdefault(x['source_image_id'], []).append(x)
    return meme_info_map, template_info_map

def save_image_with_added_text(meme_info_list, template_info, data_root_dir, font_path):
    image = Image.open(os.path.join(data_root_dir, 'source_images', '{}.jpg'.format(template_info['source_image_id'])))
    for i, row in enumerate(meme_info_list):
        image_copy = image.copy()
        texts = row['texts']
        for box in template_info['boxes']:
            text = texts[box['label']].strip()
            add_text_in_box(box['points'], text, image_copy, font_path)
        image_save_path = os.path.join(os.path.join(data_root_dir, 'full_images'), row['image_name'])
        image_copy.save(image_save_path)

def main():
    meme_info_map, template_info_map = read_meta_data()
    pool = Pool(processes=NUM_WORKER)
    process_bar = tqdm(total=len(template_info_map))

    for source_image_id, meme_info_list in meme_info_map.items():
        pool.apply_async(save_image_with_added_text,
                         args=(meme_info_list, template_info_map[source_image_id],
                               DATA_ROOT_DIR, FONT_PATH),
                         callback=lambda *args: process_bar.update())

    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
