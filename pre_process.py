import re

def split_data():
    target_file = open(file='./data/tweet_proc.csv', mode='w', encoding='utf-8')
    expression_file = open(file='./data/tweet_expression.csv', mode='w', encoding='utf-8')
    tweet_original = open(file='./data/tweet_original.csv', mode='w', encoding='utf-8')
    expression_pattern = re.compile('.*\[.*\].*')
    with open(file='./data/weibo_tweet2.txt', mode='r', encoding='utf-8') as f:
            for line in f:
                new_line = line.replace('\t', ' ').strip()
                target_file.write(new_line + '\n')
                if expression_pattern.match(new_line):
                    expression_file.write(new_line + '\n')

    target_file.close()


def split_place(content):
    if '·' in content:
        c = content.split(' ')
        if '·' in c[-1]:
            content = content.replace(c[-1], ' ')
            content += '\n'
    return content.replace(' ', '')


def parse(content, emoji=False):
    content += ' '
    at_pattern = re.compile('@(.*?)\s')
    topic_pattern = re.compile('#(.*?)#')
    colon_pattern = re.compile(':(.*?):')
    emoji_pattern = re.compile('\[(.*?)\]')

    ats = at_pattern.findall(content)
    topics = topic_pattern.findall(content)
    colons = colon_pattern.findall(content)

    if ats:
        for at in ats:
            content = content.replace('@'+at, '')
    if topics:
        for topic in topics:
            content = content.replace('#'+topic+'#', '')
    if colons:
        for colon in colons:
            content = content.replace(':' + colon + ':', '')
    if emoji:
        emojis = emoji_pattern.findall(content)
        if emojis:
            for e in emojis:
                content = content.replace('[' + e + ']', '')
    return content.strip()


def parse_cl(name='negative'):
    target = open('data/tweet_%s5-2.csv' % name, 'w', encoding='utf-8')
    with open('data/tweet_%s3.csv' % name, 'r', encoding='utf-8') as f:
        for line in f:
            if '秒拍视频' in line or '微博视频' in line\
                    or '——' in line or '互粉' in line \
                    or '微信' in line or '单子' in line or '代理' in line\
                    or '优惠券' in line or '下单' in line or '关注' in line\
                    or '转发' in line or '抽奖' in line or '圣诞' in line \
                    or '100100100100' in line or '购买' in line or '二维码' in line\
                    or '红包' in line or '接单' in line or '微博故事' in line\
                    or '分享单曲' in line or '头条文章' in line or '生日快乐' in line\
                    or '订单' in line or '饭拍' in line or '直播' in line\
                    or '分享' in line or 'sdcvfdgf' in line or 'xinqitiansh' in line\
                    or '互关' in line or '私信' in line or '活动' in line or '加薇' in line:
                continue
            # if line.strip():
            #     target.write(line.strip() + '\n')
            new_line = parse(split_place(line), False).replace('秒拍视频.', '')
            if new_line:
                target.write(new_line + '\n')
    target.close()

parse_cl()
parse_cl(name='positive')


def split_emoji():
    reply_pattern = re.compile('回复@.*?:(.*)')
    forward_pattern = re.compile('@.*?:(.*)')
    emoji_pattern = re.compile('\[(.*?)\]')

    def load_emoji(file_name):
        emoji = set()
        with open(file=file_name, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                emoji.add(line[1:-1])
        return emoji

    emoji_positive = load_emoji('./data/emoji_positive')
    emoji_negative = load_emoji('./data/emoji_negative')
    positive_file = open(file='./data/tweet_positive.csv', mode='w', encoding='utf-8')
    negative_file = open(file='./data/tweet_negative.csv', mode='w', encoding='utf-8')
    with open(file='./data/tweet_expression.csv', encoding='utf-8') as f:
        for line in f:
            for w in line.split('//'):
                w = w.strip()
                print(w)
                reply = reply_pattern.findall(w)
                if reply:
                    content = reply[0]
                else:
                    c = forward_pattern.findall(w)
                    if c:
                        content = c[0]
                    else:
                        content = w
                print(type(content))
                print(content)
                emojis = emoji_pattern.findall(content)
                negative_cnt = 0
                positive_cnt = 0
                for emoji in emojis:
                    if emoji in emoji_negative:
                        negative_cnt += 1
                    elif emoji in emoji_positive:
                        positive_cnt += 1
                if positive_cnt > negative_cnt:
                    positive_file.write(parse(content) + '\n')
                elif negative_cnt > positive_cnt:
                    negative_file.write(parse(content) + '\n')
    negative_file.close()
    positive_file.close()


