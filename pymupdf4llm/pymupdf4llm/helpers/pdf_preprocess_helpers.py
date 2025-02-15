import os
import fitz
import requests

font_name = "fangsong"
font = fitz.Font(font_name,  os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/simfang.ttf"))
concurrency = os.getenv("PDF_PARSE_CONCURRENCY", 10)


def save_page_as_image(args):
    pdf_path, page_id, page_path, dpi = args
    pdf_document = fitz.open(pdf_path)  # Open the PDF in the worker process
    page = pdf_document.load_page(page_id)
    pix = page.get_pixmap(dpi=256, annots=False)
    pix.save(page_path)
    return page_id

def extract_text_from_image(img_str):
    url = os.getenv("OCR_HOST", "http://14.22.89.198:3402/ocr")  # 服务URL
    data = {"use_det": True, "use_cls": False, "use_rec": True, 'image_data': img_str}

    response = requests.post(url, data=data, timeout=6000)
    if response.status_code != 200:
        return {'text': [], "error": response.text, "status_code": response.status_code}

    result = [(v['rec_txt'], v['dt_boxes']) for k, v in response.json().items()]
    return {"text": result, "status_code": response.status_code}


def run(pdf_document, page_num, sign, empty, font):
    import numpy as np
    import cv2
    import re

    page = pdf_document.load_page(page_num)
    page_ocred_text_cache = []
    coords_pre = -100, -100, -100, -100
    # img_idx = 1000 * page_num

    def flags_decomposer(flags):
        """Make font flags human readable."""
        l = []
        if flags & 2 ** 0:
            l.append("superscript")
        if flags & 2 ** 1:
            l.append("italic")
        if flags & 2 ** 2:
            l.append("serifed")
        else:
            l.append("sans")
        if flags & 2 ** 3:
            l.append("monospaced")
        else:
            l.append("proportional")
        if flags & 2 ** 4:
            l.append("bold")
        return ", ".join(l)

    blocks = page.get_text("dict", flags=11)["blocks"]
    for b in blocks:  # iterate through the text blocks
        for l in b["lines"]:  # iterate through the text lines
            for s in l["spans"]:  # iterate through the text spans
                font_properties = "Font: '%s' (%s), size %g, color #%06x" % (
                    s["font"],  # font name
                    flags_decomposer(s["flags"]),  # readable font flags
                    s["size"],  # font size
                    s["color"],  # font color
                )
                # print(font_properties)

    page.insert_font(fontname=font_name, fontbuffer=font.buffer)
    pix = page.get_pixmap(dpi=72 * 3, annots=False)
    image_data = np.frombuffer(pix.samples, dtype=np.uint8)
    image = image_data.reshape((pix.height, pix.width, pix.n))
    if pix.n > 1:
        scene = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        scene = image

    result_sign = cv2.matchTemplate(scene, sign, cv2.TM_CCOEFF_NORMED)
    result_empty = cv2.matchTemplate(scene, empty, cv2.TM_CCOEFF_NORMED)

    threshold = 0.9
    locations_sign = np.where(result_sign >= threshold)  # Find all locations with confidence >= 0.95
    locations_empty = np.where(result_empty >= threshold)  # Find all locations with confidence >= 0.95

    margin = 0

    h, w = sign.shape
    inserted = []

    for pt in zip(*locations_sign[::-1]):
        top_left = pt
        bottom_right = (top_left[0] + w, top_left[1] + h)
        skip = False
        for rect in inserted:
            if rect[0][0] - margin <= top_left[0] <= rect[1][0] + margin and rect[0][1] - margin <= top_left[1] <= rect[1][1] + margin:
                skip = True
            if rect[0][0] - margin <= bottom_right[0] <= rect[1][0] + margin and rect[0][1] - margin <= bottom_right[1] <= rect[1][1] + margin:
                skip = True
            if top_left[0] - margin <= rect[0][0] <= bottom_right[0] + margin and top_left[1] - margin <= rect[0][1] <= bottom_right[1] + margin:
                skip = True
            if top_left[0] - margin <= rect[1][0] <= bottom_right[0] + margin and top_left[1] - margin <= rect[1][1] <= bottom_right[1] + margin:
                skip = True

        if skip:
            continue
        # cv2.rectangle(scene_color, top_left, bottom_right, (0, 0, 255), 2)
        page.insert_text([top_left[0] / 3, bottom_right[1] / 3], '■', fontname=font_name, fontsize=11, color=[0, 0, 1])
        inserted.append((top_left, bottom_right))

    h, w = empty.shape
    inserted = []
    for pt in zip(*locations_empty[::-1]):  # Swap x and y from np.where
        top_left = pt
        bottom_right = (top_left[0] + w, top_left[1] + h)
        skip = False
        for rect in inserted:
            if rect[0][0] - margin <= top_left[0] <= rect[1][0] + margin and rect[0][1] - margin <= top_left[1] <= rect[1][1] + margin:
                skip = True
            if rect[0][0] - margin <= bottom_right[0] <= rect[1][0] + margin and rect[0][1] - margin <= bottom_right[1] <= rect[1][1] + margin:
                skip = True
            if top_left[0] - margin <= rect[0][0] <= bottom_right[0] + margin and top_left[1] - margin <= rect[0][1] <= bottom_right[1] + margin:
                skip = True
            if top_left[0] - margin <= rect[1][0] <= bottom_right[0] + margin and top_left[1] - margin <= rect[1][1] <= bottom_right[1] + margin:
                skip = True

        if skip:
            continue
        page.insert_text([top_left[0] / 3, bottom_right[1] / 3], '□', fontname=font_name, fontsize=11, color=[0, 1, 0])
        inserted.append((top_left, bottom_right))

    image_list = page.get_images(full=True)
    for img_index, img in enumerate(image_list):

        xref = img[0]  # Image reference number
        base_image = pdf_document.extract_image(xref)  # Extract image data
        image_bytes = base_image["image"]
        # page.delete_image(img[0])

        m = 4

        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        height, width = image.shape
        new_height = height * m
        new_width = width * m
        new_image = np.ones((new_height, new_width), dtype=np.uint8) * 255  # White background

        # Place the original image in the center of the new image
        x_offset = (new_width - width) // 2
        y_offset = (new_height - height) // 2
        new_image[y_offset:y_offset + height, x_offset:x_offset + width] = image
        _, new_image_bytes = cv2.imencode('.jpg', new_image)

        # # Display or save the image with the border
        # cv2.imshow("Image with Border", new_image)
        # cv2.waitKey(3)
        # cv2.destroyAllWindows()
        #
        # # To save the image:
        # img_idx += 1
        # cv2.imwrite(f"/tmp/1211/{page_num}-{img_idx}.jpg", new_image)

        image_retrived = page.get_image_rects(xref)
        if not image_retrived:
            continue

        img_rect = image_retrived[0]
        coords = (img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1)

        import base64
        texts_ocred = extract_text_from_image(base64.b64encode(new_image_bytes))['text']
        if 2 * coords[1] <= coords_pre[1] + coords_pre[3]:
            page_ocred_text_cache[-1].append((texts_ocred, coords))
        else:
            page_ocred_text_cache.append([(texts_ocred, coords)])

        coords_pre = coords

    for text_group in page_ocred_text_cache:
        total_len = 0
        total_cnt = 0

        for texts_ocred, coords in text_group:
            for rec_txt, dt_boxes in texts_ocred:
                for char in rec_txt:
                    if re.match(r'[\u3040-\u9fff]', char):
                        total_cnt += 1.0
                    else:
                        total_cnt += 0.75
            total_len += coords[2] - coords[0]

        if total_cnt == 0:
            continue

        fontsize = total_len / total_cnt / 1.02

        y_offset = 0
        for texts_ocred, coords in text_group:
            y_offset += 8
            x_offset = 0
            for rec_txt, dt_boxes in texts_ocred:
                if x_offset > 400:
                    x_offset = 0
                    y_offset += 8

                page.insert_text((coords[0] + x_offset, coords[1] + y_offset), rec_txt, fontname=font_name, fontsize=fontsize, color=[0, 0, 1])
                # print(f"inserted ocr text : {rec_txt}")
                x_offset += 8 * len(rec_txt)


def file_recently_ocred(file_path):
    import time
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist.")
        return False

    current_time = time.time()
    file_mod_time = os.path.getmtime(file_path)

    if current_time - file_mod_time <= 300:
        print(f"File '{file_path}' was updated within the last 5 minutes. skip ocr")
        return True
    else:
        print(f"File '{file_path}' was not updated within the last 5 minutes. will ocr")
        return False

def pre_process_pdf(pdf_path="/opt/maxkb/app/uploads/15c709be-b76c-11ef-9850-0242ac110002/j1209招标文件/j1209招标文件.pdf",
                    output_path="/opt/maxkb/app/uploads/tmp/j1209招标文件.pdf",
                    data_path = "/opt/maxkb/app/apps/function_lib/data/pdf_preprocess/"):
    result = {'cache': False}

    if file_recently_ocred(output_path):
        result['success'] = True
        result['cache'] = True
        return result

    import os
    import fitz
    import cv2

    sign_path = os.path.join(data_path, "sign_no_zoom.png")
    empty_path = os.path.join(data_path, "empty_no_zoom.png")

    sign = cv2.imread(sign_path, cv2.IMREAD_GRAYSCALE)
    empty = cv2.imread(empty_path, cv2.IMREAD_GRAYSCALE)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        pdf_document = fitz.open(pdf_path)

        threads = []
        for page_num in range(len(pdf_document)):
            import threading
            t = threading.Thread(target=run, args=(pdf_document, page_num, sign, empty, font))
            threads.append(t)
            t.start()

            if page_num >= concurrency:
                threads[-concurrency].join()

            # if page_num >= 12:
            #     break
        for t in threads:
            t.join()

        pdf_document.save(output_path)
        result['success'] = True
    except Exception as e:
        result['debug'] = f"Fail in image extraction: {str(e)}"
        result['success'] = False
        import traceback
        traceback.print_exc()

    return result


# pre_process_pdf("/Users/kaijie.qin/repo/biaoxin/RAG/data/nbxc.pdf", "/Users/kaijie.qin/repo/biaoxin/RAG/data/nbxc.fixed.pdf", '/Users/kaijie.qin/repo/biaoxin/RAG/data')
# pre_process_pdf("/Users/kaijie.qin/Downloads/1224_4.pdf", "/tmp/1224_4.pdf", '/Users/kaijie.qin/repo/biaoxin/RAG/data')


# import os
# for root, dirs, files in os.walk('/Volumes/usb-disk/tmp/pdf_zbwj_jinrun'):
#     for file in files:
#         if file.startswith("."):
#             continue
#         if os.path.exists(os.path.join(root+"_res", file)):
#             continue
#
#         print(pre_process_pdf(os.path.join(root, file), os.path.join(root+"_res", file), '/Users/kaijie.qin/repo/biaoxin/RAG/data'))
