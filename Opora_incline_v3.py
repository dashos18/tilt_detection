import numpy as np
import cv2
import math
import os

class Opora_incline:

    def __init__(self):
        pass

    def process(self,path_to_image,line_thickness=5):

        self.path_to_image=path_to_image
        print(self.path_to_image)
        self.mask_preproceesing(path_to_image)
        self.process_lines(path_to_image)
        #for x in self.final_lines:
            #cv2.line(self.img, (x[0][0], x[0][1]), (x[1][0], x[1][1]), (0, 0, 255), line_thickness)

        outpath='/Users/dariavolkova/Desktop/lab_future/0_DEFECTS_DETECTION/new_3_not_cut'
        image_name = os.path.split(self.path_to_image)[-1]

        cv2.imwrite(os.path.join(outpath, image_name), self.img)
        self.pill_extract()


    def mask_preproceesing(self, image):
        self.img=cv2.imread(image)
        mask=np.zeros(self.img.shape[:2],np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        rect = (20, 0, self.img.shape[1] - 20, self.img.shape[0])  # (start_x, start_y, width, height).
        cv2.grabCut(self.img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = self.img * mask2[:, :, np.newaxis]


        ret, thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        #cv2.imwrite('/Users/dariavolkova/Desktop/pred/lines_lines.jpg', thresh1)

        return thresh1

    def get_lines(self,lines_in):
        if cv2.__version__ < '3.0':
            return lines_in[0]
        return [l[0] for l in lines_in]

    def process_lines(self,img):

        img = self.mask_preproceesing(img)

        edges = cv2.Canny(img, threshold1=50, threshold2=200, apertureSize=3)

        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=100)

        # merge lines

        # ------------------
        # prepare
        old_lines = []
        angle_thresh = 60
        for _line in self.get_lines(lines):
            old_lines.append([(_line[0], _line[1]), (_line[2], _line[3])])

        #print('This is all lines',old_lines)

        _lines = []

        for line in old_lines:
            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[1][0]
            y2 = line[1][1]

            angle = abs(round(np.rad2deg(np.arctan2((y2 - y1), (x2 - x1))), 2))
            if angle < angle_thresh:
                continue

            _lines.append([(line[0][0], line[0][1]), (line[1][0], line[1][1])])

        # pint('This is new_lines', len(new_lines))

        # sort and get orientation of line
        _lines_x = []
        _lines_y = []
        for line_i in _lines:
            orientation_i = math.atan2((line_i[0][1] - line_i[1][1]), (line_i[0][0] - line_i[1][0]))
            if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90 + 45):
                _lines_y.append(line_i)
            else:
                _lines_x.append(line_i)

        _lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
        _lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])

        merged_lines_x = self.merge_lines_pipeline_2(_lines_x)
        merged_lines_y = self.merge_lines_pipeline_2(_lines_y)
        #print('X',merged_lines_x, 'Y',merged_lines_y)

        merged_lines_all = []
        merged_lines_all.extend(merged_lines_x)
        merged_lines_all.extend(merged_lines_y)
        #print("process groups lines", len(_lines), 'All important lines',len(merged_lines_all))

        # merged_lines_all.sort(key=lambda x: x[1])
        # new_something = []
        # for i in range(len(merged_lines_all) - 1):
        #     if abs(merged_lines_all[i][1][0] - merged_lines_all[i + 1][0][0]) <= 5:
        #         new_something.append(merged_lines_all[i][0])
        #         new_something.append(merged_lines_all[i + 1][1])
        #     else:
        #         merged_lines_all.extend(new_something)

        #img_merged_lines = img
        #for line in merged_lines_all:
            #cv2.line(img_merged_lines, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (0, 0, 255), 6)
            #cv2.imwrite('/Users/dariavolkova/Desktop/pred/lines_lines.jpg', img)

        #self.merge_lines_all=merged_lines_all

        two_lines = []
        dop_line = [(int(img.shape[1] / 2), 0), (int(img.shape[1] / 2),img.shape[0])]
        for line in merged_lines_all:
            if line[0][0] <= dop_line[0][0]:
                two_lines.append(line)
                break
        for line in merged_lines_all:
            if line[0][0] > dop_line[0][0]:
                two_lines.append(line)
                break

        # find length of the first line
        length_l1 = math.sqrt(
            (two_lines[0][0][0] - two_lines[0][1][0]) ** 2 + (two_lines[0][0][1] - two_lines[0][1][1]) ** 2)
        # find new coordinate:
        # when y=0
        x1 = int(round(
            two_lines[0][0][0] + (two_lines[0][0][0] - two_lines[0][1][0]) / length_l1 * two_lines[0][0][1]))
        # when y=img.shape[0]
        x1_1 = int(round(two_lines[0][1][0] + (two_lines[0][1][0] - two_lines[0][0][0]) / length_l1 * (
                    img.shape[0] - two_lines[0][1][1])))

        length_l2 = math.sqrt(
            (two_lines[1][0][0] - two_lines[1][1][0]) ** 2 + (two_lines[1][0][1] - two_lines[1][1][1]) ** 2)
        # when y=0
        x2 = int(round(
            two_lines[1][0][0] + (two_lines[1][0][0] - two_lines[1][1][0]) / length_l2 * two_lines[1][0][1]))
        # when y=img.shape[0]
        x2_2 = int(round(two_lines[1][1][0] + (two_lines[1][1][0] - two_lines[1][0][0]) / length_l2 * (
                    img.shape[0] - two_lines[1][1][1])))


        final_lines = []
        # final_lines.append([[x1, 0]] + [[x1_1, img.shape[0]]])
        # final_lines.append([[x2, 0]] + [[x2_2, img.shape[0]]])
        final_lines.append([x1, 0])
        final_lines.append([x1_1, img.shape[0]])
        final_lines.append([x2, 0])
        final_lines.append([x2_2, img.shape[0]])

        self.final_lines=final_lines
        print('THIS IS TWO FINAL LINES',self.final_lines)

        return final_lines

    def pill_extract(self):
        border_lines=self.final_lines
        last_point=[self.final_lines[2]]
        print(last_point)
        pts=np.array(border_lines+last_point)
        print('This is new coordinate',pts)
        mask = np.zeros((self.img.shape[0], self.img.shape[1]))
        cv2.fillConvexPoly(mask, pts, 1)
        mask = mask.astype(np.bool)
        out = np.zeros_like(self.img)
        out[mask] = self.img[mask]

        outpath = '/Users/dariavolkova/Desktop/kk'
        image_name = os.path.split(self.path_to_image)[-1]
        cv2.imwrite(os.path.join(outpath, image_name), out)

        #array of only pillar
        return out[mask]

    def merge_lines_pipeline_2(self,lines):
        super_lines_final = []
        super_lines = []
        min_distance_to_merge = 30 #was 30
        min_angle_to_merge = 30 #was 30

        #check if line have angle and enough distance to be similar

        for line in lines:
            create_new_group = True
            group_updated = False

            for group in super_lines:
                for line2 in group:
                    if self.get_distance(line2, line) < min_distance_to_merge:
                        # check the angle between lines
                        orientation_i = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))
                        orientation_j = math.atan2((line2[0][1] - line2[1][1]), (line2[0][0] - line2[1][0]))

                        if int(abs(abs(math.degrees(orientation_i)) - abs(
                                math.degrees(orientation_j)))) < min_angle_to_merge:
                            # print("angles", orientation_i, orientation_j)
                            # print(int(abs(orientation_i - orientation_j)))
                            group.append(line)

                            create_new_group = False
                            group_updated = True
                            break

                if group_updated:
                    break

            if (create_new_group):
                new_group = []
                new_group.append(line)

                for idx, line2 in enumerate(lines):
                    # check the distance between lines
                    if self.get_distance(line2, line) < min_distance_to_merge:
                        # check the angle between lines
                        orientation_i = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))
                        orientation_j = math.atan2((line2[0][1] - line2[1][1]), (line2[0][0] - line2[1][0]))

                        if int(abs(abs(math.degrees(orientation_i)) - abs(
                                math.degrees(orientation_j)))) < min_angle_to_merge:
                            # print("angles", orientation_i, orientation_j)
                            # print(int(abs(orientation_i - orientation_j)))

                            new_group.append(line2)

                            # remove line from lines list
                            # lines[idx] = False
                # append new group
                super_lines.append(new_group)

        for group in super_lines:
            super_lines_final.append(self.merge_lines_segments1(group))

        return super_lines_final

    def merge_lines_segments1(self,lines, use_log=False):
        if (len(lines) == 1):
            return lines[0]

        line_i = lines[0]

        # orientation
        orientation_i = math.atan2((line_i[0][1] - line_i[1][1]), (line_i[0][0] - line_i[1][0]))

        points = []
        for line in lines:
            points.append(line[0])
            points.append(line[1])

        if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90 + 45):

            # sort by y
            points = sorted(points, key=lambda point: point[1])

            if use_log:
                print("use y")
        else:

            # sort by x
            points = sorted(points, key=lambda point: point[0])

            if use_log:
                print("use x")

        return [points[0], points[len(points) - 1]]

    def lines_close(self,line1, line2):
        dist1 = math.hypot(line1[0][0] - line2[0][0], line1[0][0] - line2[0][1])
        dist2 = math.hypot(line1[0][2] - line2[0][0], line1[0][3] - line2[0][1])
        dist3 = math.hypot(line1[0][0] - line2[0][2], line1[0][0] - line2[0][3])
        dist4 = math.hypot(line1[0][2] - line2[0][2], line1[0][3] - line2[0][3])

        if (min(dist1, dist2, dist3, dist4) < 100):
            return True
        else:
            return False

    def lineMagnitude(self,x1, y1, x2, y2):
        lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
        return lineMagnitude


    def DistancePointLine(self,px, py, x1, y1, x2, y2):
        # http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        LineMag = self.lineMagnitude(x1, y1, x2, y2)

        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            # // closest point does not fall within the line segment, take the shorter distance
            # // to an endpoint
            ix = self.lineMagnitude(px, py, x1, y1)
            iy = self.lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = self.lineMagnitude(px, py, ix, iy)

        return DistancePointLine

    def get_distance(self,line1, line2):
        dist1 = self.DistancePointLine(line1[0][0], line1[0][1],
                                  line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        dist2 = self.DistancePointLine(line1[1][0], line1[1][1],
                                  line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        dist3 = self.DistancePointLine(line2[0][0], line2[0][1],
                                  line1[0][0], line1[0][1], line1[1][0], line1[1][1])
        dist4 = self.DistancePointLine(line2[1][0], line2[1][1],
                                  line1[0][0], line1[0][1], line1[1][0], line1[1][1])

        return min(dist1, dist2, dist3, dist4)



basepath='/Users/dariavolkova/Desktop/lab_future/0_DEFECTS_DETECTION/dataset_opora'
processer=Opora_incline()

for image in os.listdir(basepath):

    if not any((image.endswith(ext) for ext in [".png", "jpg"])):
        continue
    testing=processer.process(os.path.join(basepath,image))



