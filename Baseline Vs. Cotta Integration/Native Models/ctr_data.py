import csv
import operator
import argparse
import random
import numpy as np
import os
import pandas as pd

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def to_time_frac(hour, minute, time_frac_dict):
    for key in time_frac_dict[hour].keys():
        if key[0] <= minute <= key[1]:
            return str(time_frac_dict[hour][key])

def to_libsvm_encode(datapath, time_frac_dict):
    print('Converting to libsvm format')
    oses = ["windows", "ios", "mac", "android", "linux"]
    browsers = ["chrome", "sogou", "maxthon", "safari", "firefox", "theworld", "opera", "ie"]

    f1s = ["weekday", "hour", "IP", "region", "city", "adexchange", "domain", "slotid", "slotwidth", "slotheight",
           "slotvisibility", "slotformat", "creative", "advertiser"]

    f1sp = ["useragent", "slotprice"]

    def feat_trans(name, content):
        # Feature transformation
        content = content.lower()
        if name == "useragent":
            operation = "other"
            for o in oses:
                if o in content:
                    operation = o
                    break
            browser = "other"
            for b in browsers:
                if b in content:
                    browser = b
                    break
            return operation + "_" + browser
        if name == "slotprice":
            price = int(content)
            if price > 100:
                return "101+"
            elif price > 50:
                return "51-100"
            elif price > 10:
                return "11-50"
            elif price > 0:
                return "1-10"
            else:
                return "0"

    def get_tags(content):
        if content == '\n' or len(content) == 0:
            return ["null"]
        return content.strip().split(',')[:5]

    # initialize
    namecol = {}  # Name-column dictionary
    featindex = {}  # Feature index dictionary
    maxindex = 0  # Maximum index

    with open(os.path.join(datapath, 'train.bid.csv'), 'r') as fi:
        first = True

        featindex['truncate'] = maxindex
        maxindex += 1

        for line in fi:
            s = line.split(',')
            if first:
                first = False
                for i in range(0, len(s)):
                    namecol[s[i].strip()] = i
                    if i > 0:
                        featindex[str(i) + ':other'] = maxindex
                        maxindex += 1
                continue
            for f in f1s:
                col = namecol[f]
                content = s[col]
                feat = str(col) + ':' + content
                if feat not in featindex:
                    featindex[feat] = maxindex
                    maxindex += 1
            for f in f1sp:
                col = namecol[f]
                content = feat_trans(f, s[col])
                feat = str(col) + ':' + content
                if feat not in featindex:
                    featindex[feat] = maxindex
                    maxindex += 1
            col = namecol["usertag"]
            tags = get_tags(s[col])
            feat = str(col) + ':' + ''.join(tags)
            if feat not in featindex:
                featindex[feat] = maxindex
                maxindex += 1

    print('Feature size: ' + str(maxindex))
    featvalue = sorted(featindex.items(), key=operator.itemgetter(1))

    with open(os.path.join(datapath, 'feat.bid.txt'), 'w') as fo:
        fo.write(str(maxindex) + '\n')
        for fv in featvalue:
            fo.write(fv[0] + '\t' + str(fv[1]) + '\n')

    # Indexing train
    print('Indexing ' + datapath + '/train.bid.csv')
    with open(os.path.join(datapath, 'train.bid.csv'), 'r') as fi, open(os.path.join(datapath, 'train.bid.txt'), 'w') as fo:
        first = True
        for line in fi:
            if first:
                first = False
                continue
            s = line.split(',')
            time_frac = s[4][8:12]
            fo.write(s[0] + ',' + s[23] + ',' + s[2] + ',' + to_time_frac(int(time_frac[0:2]), int(time_frac[2:4]), time_frac_dict) + ',' + str(s[4]))
            index = featindex['truncate']
            fo.write(',' + str(index))
            for f in f1s:
                col = namecol[f]
                content = s[col]
                feat = str(col) + ':' + content
                if feat not in featindex:
                    feat = str(col) + ':other'
                index = featindex[feat]
                fo.write(',' + str(index))
            for f in f1sp:
                col = namecol[f]
                content = feat_trans(f, s[col])
                feat = str(col) + ':' + content
                if feat not in featindex:
                    feat = str(col) + ':other'
                index = featindex[feat]
                fo.write(',' + str(index))
            col = namecol["usertag"]
            tags = get_tags(s[col])
            feat = str(col) + ':' + ''.join(tags)
            if feat not in featindex:
                feat = str(col) + ':other'
            index = featindex[feat]
            fo.write(',' + str(index))
            fo.write('\n')

    # Indexing test
    print('Indexing ' + datapath + '/test.bid.csv')
    with open(os.path.join(datapath, 'test.bid.csv'), 'r') as fi, open(os.path.join(datapath, 'test.bid.txt'), 'w') as fo:
        first = True
        for line in fi:
            if first:
                first = False
                continue
            s = line.split(',')
            time_frac = s[4][8:12]
            fo.write(s[0] + ',' + s[23] + ',' + s[2] + ',' + to_time_frac(int(time_frac[0:2]), int(time_frac[2:4]), time_frac_dict) + ',' + str(s[4]))
            index = featindex['truncate']
            fo.write(',' + str(index))
            for f in f1s:
                col = namecol[f]
                if col >= len(s):
                    print('col: ' + str(col))
                    print(line)
                content = s[col]
                feat = str(col) + ':' + content
                if feat not in featindex:
                    feat = str(col) + ':other'
                index = featindex[feat]
                fo.write(',' + str(index))
            for f in f1sp:
                col = namecol[f]
                content = feat_trans(f, s[col])
                feat = str(col) + ':' + content
                if feat not in featindex:
                    feat = str(col) + ':other'
                index = featindex[feat]
                fo.write(',' + str(index))
            col = namecol["usertag"]
            tags = get_tags(s[col])
            feat = str(col) + ':' + ''.join(tags)
            if feat not in featindex:
                feat = str(col) + ':other'
            index = featindex[feat]
            fo.write(',' + str(index))
            fo.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/vladplyusnin/tftest/Deep-Learning-COPSCI764/Project/data')
    parser.add_argument('--dataset_name', default='ipinyou', help='ipinyou')
    parser.add_argument('--campaign_id', default='total', help='e.g. total')
    parser.add_argument('--is_to_csv', default=True)

    setup_seed(1)

    args = parser.parse_args()
    print("Dataset: " + args.campaign_id)
    data_path = os.path.join(args.data_path, args.dataset_name, args.campaign_id)

    # Create directory if it does not exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Segment into 96 time slots (15 minutes each)
    time_frac_dict = {}
    count = 0
    for i in range(24):
        hour_frac_dict = {}
        for item in [(0, 15), (15, 30), (30, 45), (45, 60)]:
            hour_frac_dict.setdefault(item, count)
            count += 1
        time_frac_dict.setdefault(i, hour_frac_dict)

    if args.is_to_csv:
        print('Converting to csv')
        with open(os.path.join(data_path, 'train.bid.csv'), 'w', newline='') as csv_file:
            spam_writer = csv.writer(csv_file, dialect='excel')
            with open(os.path.join(data_path, 'train.log.txt'), 'r') as filein:
                for i, line in enumerate(filein):
                    line_list = line.strip('\n').split('\t')
                    spam_writer.writerow(line_list)
        print('Train data read/write complete')

        with open(os.path.join(data_path, 'test.bid.csv'), 'w', newline='') as csv_file:
            spam_writer = csv.writer(csv_file, dialect='excel')
            with open(os.path.join(data_path, 'test.log.txt'), 'r') as filein:
                for i, line in enumerate(filein):
                    line_list = line.strip('\n').split('\t')
                    spam_writer.writerow(line_list)
        print('Test data read/write complete')

    to_libsvm_encode(data_path, time_frac_dict)
    os.remove(os.path.join(data_path, 'train.bid.csv'))
    os.remove(os.path.join(data_path, 'test.bid.csv'))
