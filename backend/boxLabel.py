import os
import json
import math

    # 중심값 리스트 (예시)

class BoxLabel():
    bin_box =[
    (847, 765, 'bs_name'),
    (1830, 764, 'bs_register_num'),
    (739, 867, 'co_adress'),
    (1219, 862, 'co_adress'),
    (1646, 866, 'co_adress'),
    (2102, 869, 'co_adress'),
    (985, 953, 'dept_name'),
    (1728, 953, 'position'),
    (970, 1031, 'manager_name'),
    (945, 1113, 'bs_call'),
    (1732, 1104, 'bs_fax'),
    (1044, 1173, 'bs_email'),(869, 1433, 'bank_type'),(1781, 1441, 'account_num'),(913, 1562, 'expected_vacc_count'),(992, 1696, 'purpose_use'),(2078, 2068, 'agree_use_only'),(2076, 2203, 'agree_misuse_action'),(2079, 2369, 'agree_inspect'),(2075, 2532, 'agree_close_if_unused'),(2076, 2688, 'agree_notify_not_owner'),(532, 2992, 'YY'),(914, 2982, 'MM'),(1145, 2989, 'dd'),(1788, 2989, 'writer_name')]


    def __init__(self):
        bin_root = "./document/labelings.json"
        if os.path.exists(bin_root):
            print("파일이 존재합니다.")
        else:
            print("파일이 존재하지 않습니다.")
        with open(bin_root, 'r', encoding='utf-8') as file:
            bin = json.load(file)
        self.labels = bin



    def get_data(self, data):
        print(data["document_info"]["document_type"])
        doc = data["document_info"]["document_type"]
        # self.bin_box = bin
        doctype_list = list(filter(lambda b: b["type_name"] == doc, self.labels))

        if not doctype_list:
            print(f"문서 타입 '{doc}'을 찾을 수 없습니다. 기본 라벨링을 사용합니다.")
            # 기본 라벨링 사용 또는 에러 반환
            return data

        doctype = doctype_list[0]
        # print(doctype)
        self.bin_box = doctype["bin_box"]
        data = self.formlabeling(data)

        self.data = data

        return self.data


    def euclidean_distance(self, p1, p2, r_width, r_height):
        return math.sqrt((p1[0]/4 - (p2[0]/4)*r_width)**2 + (p1[1] - (p2[1])*r_height)**2)

    # 결과 저장


    #신청서
    #bin_box = [(759, 444, 'bs_name'), (2097, 442, 'bs_register_num'), (699, 533, 'bs_name'), (1096, 529, 'bs_name'), (2084, 532, 'co_register_num'), (725, 625, 'co_adress'), (1130, 623, 'co_adress'), (1569, 623, 'co_adress'), (1947, 620, 'co_adress'), (765, 712, 'bs_call'), (2035, 713, 'bs_fax'), (510, 801, 'ceo_kor_name'), (889, 804, 'ceo_eng_name'), (1243, 798, 'ceo_eng_name'), (1977, 802, 'ceo_birth'), (564, 891, 'ceo_res'), (871, 891, 'ceo_res'), (1186, 892, 'ceo_res'), (1459, 890, 'ceo_res'), (2038, 887, 'ceo_call'), (581, 1070, 'pay_inis'), (2052, 1065, 'pay_account'), (604, 1150, 'credit_pay_day'), (809, 1229, 'co_credit_pay_limit'), (814, 1295, 'sect_credit_pay_limit'), (980, 1396, 'sub_email'), (713, 1492, 'pay_sub_b'), (203, 3245, 'YY'), (385, 3249, 'MM'), (596, 3241, 'dd'), (1349, 3246, 'bs_name'), (1944, 3245, 'ceo_sign')]
    # bin_box = [(939, 1838, 'account_num'), (2046, 1839, 'cust_name'), (1160, 2252, 'before_by_type'), (1577, 2253, 'before_by_type'), (1949, 2250, 'before_by_type'), (740, 2360, 'after_by_type'), (1152, 2361, 'after_by_type'), (1575, 2361, 'after_by_type'), (1965, 2358, 'after_by_type'), (1128, 2721, 'YY'), (1252, 2723, 'MM'), (1375, 2721, 'dd'), (684, 2822, 'cust_adress'), (949, 2820, 'cust_adress'), (1240, 2821, 'cust_adress'), (1492, 2815, 'cust_adress'), (1935, 2817, 'cust_call'), (1022, 2919, 'cust_c_regist_num'), (1933, 2975, 'cust_sign')]


    def square_sym(self, square):
        return (int((square[0][0] + square[1][0] + square[2][0] + square[3][0]) / 4), int((square[0][1] + square[1][1] + square[2][1] + square[3][1]) / 4), '임시')

    def boxLists(self, data):
        bboxes = []
        for box in data['fields']:
            x1, x2, x3, x4 = box['value_box']['x']
            y1, y2, y3, y4 = box['value_box']['y']
            square = [[x1, y1],[x2, y2],[x3, y3],[x4, y4]]
            r_box = self.square_sym(square)
            bboxes.append(r_box)
        return bboxes
    # H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0000003.json
    # H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0000103.json

    def getBoxLotation(self, x_los, y_los):
        x1, x2, x3, x4 = x_los[0], x_los[1], x_los[2], x_los[3]
        y1, y2, y3, y4 = y_los[0], y_los[1], y_los[2], y_los[3]

        square = [[x1, y1],[x2, y2],[x3, y3],[x4, y4]]
        r_box = self.square_sym(square)
        closest_b = min(self.bin_box, key=lambda b: self.euclidean_distance(r_box, b, 1, 1))
        # print(closest_b)
        return closest_b[2]


    def formlabeling(self, data):
        box_width, box_height = 2480, 3508
            
        # ((x1 + x2) / 2, (y1 + y2) / 2)
        bboxes = self.boxLists(data)
        b_box_width = data["document_info"]["width"]
        b_box_height = data["document_info"]["height"]
        r_width = b_box_width/box_width
        r_height = b_box_height/box_height
        print("===============",r_width, r_height)

        closest_b_for_a = []
        index = 0

        for a in bboxes:
            closest_b = min(self.bin_box, key=lambda b: self.euclidean_distance(a, b, r_width, r_height))
            a_box_label = [a[0],a[1],closest_b[2]]
            data["fields"][index]["labels"] = closest_b[2]
            
            # print(data['fields'][index]['value_text'])
            # a[2] = closest_b[2]
            closest_b_for_a.append(a_box_label)
            index+= 1
        # for a in closest_b_for_a:
        #     print(f"B 중심 {a}")

        return data


    # if __name__ == "__main__":
    #     # file_roots = ['H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0000103.json',
    #     #               'H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0000203.json',
    #     #               'H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0000303.json',
    #     #               'H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0000403.json',
    #     #               'H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0000603.json',
    #     #               'H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0000703.json',
    #     #               'H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0000803.json',
    #     #               'H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0000903.json',
    #     #               'H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0001003.json',
    #     #               'H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0001303.json',
    #     #               'H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0001403.json',
    #     #               'H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0001603.json',
    #     #               'H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0001703.json',
    #     #               'H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0001803.json',
    #     #               'H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0001903.json',
    #     #               'H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0002003.json',
    #     #               'H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0002103.json',
    #     #               'H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-2\IMG_OCR_6_F_0002203.json']
    #     # file_roots = ['.\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-1\IMG_OCR_6_F_0000102.json',
    #     #               '.\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-1\IMG_OCR_6_F_0000202.json']
    #     file_roots = ['H:\proflowworkspace\data\OCR_data_p\\realdata_uh\Training\zlabel\\1-3\IMG_OCR_6_F_0000105.json']
    #     for file_root in file_roots:
    #         box_width, box_height = 2480, 3508
            
    #         if os.path.exists(file_root):
    #             print("파일이 존재합니다.")
    #         else:
    #             print("파일이 존재하지 않습니다.")

    #         with open(file_root, 'r', encoding='utf-8') as file:
    #             data = json.load(file)
                
    #         # # ((x1 + x2) / 2, (y1 + y2) / 2)
    #         bboxes = boxLists(data)
    #         # print(bboxes)
            
    #         # bboxes = boxLists(file_root)
            
    #         bboxes = formlabeling(file_root)
    #         print(bboxes)

#conda activate mypytorch
#python boxLabel.py