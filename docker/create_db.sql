-- phpMyAdmin SQL Dump
-- version 5.1.3
-- https://www.phpmyadmin.net/
--
-- 主機： 127.0.0.1
-- 產生時間： 2022-08-15 19:07:52
-- 伺服器版本： 10.4.24-MariaDB
-- PHP 版本： 8.1.5

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";

--
-- 資料庫: `nl2sql`
--
CREATE DATABASE IF NOT EXISTS `nl2sql` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `nl2sql`;

-- --------------------------------------------------------

--
-- 資料表結構 `bank`
--

DROP TABLE IF EXISTS `bank`;
CREATE TABLE `bank` (
  `银行` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `本周涨跌幅` float DEFAULT NULL,
  `本月涨跌幅` float DEFAULT NULL,
  `年初至今涨跌幅` float DEFAULT NULL,
  `本周相对大盘收益` float DEFAULT NULL,
  `本月相对大盘收益` float DEFAULT NULL,
  `年初至今相对大盘收益` float DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- 資料表新增資料前，先清除舊資料 `bank`
--

TRUNCATE TABLE `bank`;
--
-- 傾印資料表的資料 `bank`
--

INSERT INTO `bank` (`银行`, `本周涨跌幅`, `本月涨跌幅`, `年初至今涨跌幅`, `本周相对大盘收益`, `本月相对大盘收益`, `年初至今相对大盘收益`) VALUES
('农业银行', 2.6, 0.8, -1.3, 3.2, 5.9, 24),
('交通银行', 2.5, -1, -2, 3.1, 4.1, 23.3),
('光大银行', 1.4, -3.9, -4.1, 2, 1.2, 21.2),
('中国银行', 1.1, 0, -4.5, 1.7, 5.1, 20.8),
('中信银行', 0.7, -3.9, -8.2, 1.4, 1.2, 17.1),
('招商银行', -0.7, -11.7, -10.3, -0.1, -6.6, 15),
('建设银行', 1.1, -4.2, -13.4, 1.7, 0.9, 11.9),
('民生银行', 0.2, -6.1, -16.9, 0.8, -1, 8.4);

-- --------------------------------------------------------

--
-- 資料表結構 `book`
--

DROP TABLE IF EXISTS `book`;
CREATE TABLE `book` (
  `书名` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `定价` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `著者` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `出版地` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `出版社` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `出版时间` float DEFAULT NULL,
  `页数` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `尺寸` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `附注` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `内容提要` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `主题` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `作者` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `读者群` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- 資料表新增資料前，先清除舊資料 `book`
--

TRUNCATE TABLE `book`;
--
-- 傾印資料表的資料 `book`
--

INSERT INTO `book` (`书名`, `定价`, `著者`, `出版地`, `出版社`, `出版时间`, `页数`, `尺寸`, `附注`, `内容提要`, `主题`, `作者`, `读者群`) VALUES
('南瓜小人', 'CNY25.00', '张洁著', '福州', '福建少年儿童出版社', 2018.08, '115页', '21cm', '彩虹桥名家散文诗系列', '本书精选张洁创作的散文诗多首，全书分在天空飞翔、童话树下、生命的印迹三个主题，收录了《飞》《一阵风吹过》《闪亮的日子》《喃喃》《南瓜小人》等作品。', '散文诗', '张洁', '本书适用于文学爱好者'),
('钢铁是怎样炼成的', 'CNY22.80', '(苏)尼·奥斯特洛夫斯基著', '西安', '陕西人民教育出版社', 2016, '266页', '21cm', '精装珍藏版', '本书通过记叙保尔·柯察金的成长道路告诉人们，一个人只有在革命的艰难困苦中战胜敌人也战胜自己，只有在把自己的追求和祖国、人民的利益联系在一起的时候，才会创造出奇迹，才会成长为钢铁战士。', '长篇小说', '奥斯特洛夫斯基', '本书适用于少年儿童'),
('水浒传', 'CNY22.80', '(元)施耐庵著', '西安', '陕西人民教育出版社', 2016, '218页', '21cm', '精装珍藏版', '本书围绕“官逼民反”这一线索展开情节，描写了一百零八位英雄好汉被逼上梁山，队伍逐渐壮大，起义造反到最后接受招安的全过程。小说成功地塑造了宋江、林冲、李逵、鲁智深、武松等人物形象，也向读者展示了宋代的政治与社会状况。', '章回小说', '施耐庵', '本书适用于少年儿童'),
('老人与海', 'CNY22.80', '(美)海明威著', '西安', '陕西人民教育出版社', 2016, '154页', '19cm', '精装珍藏版', '本书写的是老渔夫圣地亚哥在海上的捕鱼经历：老人制服大马林鱼后，在返航途中又同鲨鱼进行惊险的搏斗。作品中的形象具有很强的象征意蕴。', '长篇小说', '海明威', '本书适用于少年儿童'),
('红与黑', 'CNY22.80', '(法)司汤达著', '西安', '陕西人民教育出版社', 2016, '234页', '21cm', '精装珍藏版', '本书主人公于连出身于小业主家庭，醉心于拿破仑丰功伟绩的他，一心希望出人头地，无奈当时的法国正处于波旁王朝复辟时期。从军无门的他选择了教会的道路，由于能够背诵整本《新约》，于连被当地市长看中，成为他家的家庭教师，后又经教会举荐，为保王党中坚人物拉莫尔侯爵担任私人秘书。但最终，一封告密信使他的飞黄腾达毁于一旦。', '长篇小说', '司汤达', '本书适用于少年儿童'),
('三国演义', 'CNY22.80', '(明)罗贯中著', '西安', '陕西人民教育出版社', 2016, '218页', '21cm', '精装珍藏版', '本书以描写战争为主，反映了蜀（汉）、魏、吴三个政治集团之间的政治和军事斗争，大致分为黄巾之乱、董卓之乱、群雄逐鹿、三国鼎立、三国归晋五大部分。', '章回小说', '罗贯中', '本书适用于少年儿童'),
('简爱', 'CNY22.80', '(英)夏洛蒂·勃朗特著', '西安', '陕西人民教育出版社', 2016, '266页', '21cm', '精装珍藏版', '本书是十九世纪英国著名的女作家夏洛蒂·勃朗特的代表作。讲述一位从小变成孤儿的英国女子在各种磨难中不断追求自由与尊严，坚持自我，最终获得幸福的故事。', '长篇小说', '勃朗特', '本书适用于少年儿童'),
('鲁滨孙漂流记', 'CNY22.80', '(英)丹尼尔·笛福著', '西安', '陕西人民教育出版社', 2016, '250页', '21cm', '精装珍藏版', '本书讲述了英国年轻的航海爱好者鲁滨逊，在一次海难中被风浪卷到一座荒岛上。虽然脱离了危险，但他孤身一人，无依无靠，为了生存，他用自己的聪慧和坚强意志，克服了种种磨难，建造了堡垒、“别墅”和船只。他又救下土人手中的俘虏，协助英国船主收复了被海盗占领的大船。最后告别了他生活了28年的荒岛，随船返回了英国。', '长篇小说', '笛福', '本书适用于少年儿童'),
('西游记', 'CNY22.80', '(明)吴承恩著', '西安', '陕西人民教育出版社', 2016.08, '202页', '21cm', '精装珍藏版', '本书是中国古代一部浪漫主义长篇神魔小说，主要描写了唐僧、孙悟空、猪悟能、沙悟净师徒四人去西天取经，历经九九八十一难最后终于取得真经的故事。', '章回小说', '吴承恩', '本书适用于少年儿童'),
('骆驼祥子', 'CNY22.80', '老舍著', '西安', '陕西人民教育出版社', 2016, '250页', '21cm', '精装珍藏版', '本书主要是以北平（今北京）一个人力车夫祥子的行踪为线索，以二十年代末期的北京市民生活为背景，以洋车夫祥子的坎坷、悲惨的生活遭遇为主要情节，深刻揭露了旧中国的黑暗，控诉了统治阶级对劳动者的剥削、压迫，表达了作者对劳动人民的深切同情，向人们展示军阀混战、黑暗统治下的北京底层贫苦市民生活于痛苦深渊中的图景。', '长篇小说', '老舍', '本书适用于室内儿童'),
('红楼梦', 'CNY22.80', '(清)曹雪芹著', '西安', '陕西人民教育出版社', 2016, '218页', '21cm', '精装珍藏版', '本书以荣国府的日常生活为中心，以宝玉、黛玉、宝钗的爱情婚姻悲剧及大观园中点滴琐事为主线，以金陵贵族名门贾、史、王、薛四大家族由鼎盛走向衰亡的历史为暗线，展现了穷途末路的封建社会终将走向灭亡的必然趋势。', '章回小说', '曹雪芹', '本书适用于少年儿童'),
('童年', 'CNY22.80', '(苏)高尔基著', '西安', '陕西人民教育出版社', 2016, '218页', '21cm', '精装珍藏版', '本书讲述了阿廖沙（高尔基的乳名）三岁到十岁这一时期的童年生活，生动地再现了19世纪七八十年代前苏联下层人民的生活状况，写出了高尔基对苦难的认识，对社会人生的独特见解，字里行间涌动着一股生生不息的热望与坚强。', '长篇小说', '高尔基', '本书适用于少年儿童');

-- --------------------------------------------------------

--
-- 資料表結構 `hospital`
--

DROP TABLE IF EXISTS `hospital`;
CREATE TABLE `hospital` (
  `序号` float DEFAULT NULL,
  `科室` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `岗位` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `职数` float DEFAULT NULL,
  `学历` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `专业及其他要求` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `备注` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- 資料表新增資料前，先清除舊資料 `hospital`
--

TRUNCATE TABLE `hospital`;
--
-- 傾印資料表的資料 `hospital`
--

INSERT INTO `hospital` (`序号`, `科室`, `岗位`, `职数`, `学历`, `专业及其他要求`, `备注`) VALUES
(2, '关节与骨病外科', '临床医生', 1, '硕士研究生及以上', '骨外科学', '三甲医院聘任副高及以上职称'),
(3, '儿童骨科', '临床医生', 1, '硕士研究生及以上', '骨外科学', '派遣台州院区'),
(4, '生殖医学中心', '临床医生', 1, '硕士研究生及以上', '生殖医学', '派遣台州院区'),
(5, '儿童感染科', '临床医生', 1, '硕士研究生及以上', '临床医学', '派遣台州院区'),
(6, '儿童急诊与重症医学科', '临床医生', 2, '硕士研究生及以上', '临床医学', '派遣台州院区'),
(8, '儿童神经科', '临床医生', 1, '硕士研究生及以上', '临床医学', '派遣台州院区'),
(9, '儿童保健科', '临床医生', 2, '硕士研究生及以上', '临床医学', '派遣台州院区'),
(10, '超声影像科', '临床医生', 4, '硕士研究生及以上', '临床医学、影像医学与核医学', '三甲医院聘任中级及以上职称'),
(11, '放射影像科', '临床医生', 2, '硕士研究生及以上', '临床医学、放射医学、影像医学与核医学', '三甲医院聘任中级及以上职称、介入岗'),
(12, '放射影像科', '临床医生', 1, '硕士研究生及以上', '影像医学与核医学', '派遣台州院区');

-- --------------------------------------------------------

--
-- 資料表結構 `stock`
--

DROP TABLE IF EXISTS `stock`;
CREATE TABLE `stock` (
  `股票代号/名称` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `成交价` float NOT NULL,
  `涨跌` float NOT NULL,
  `涨跌幅` float NOT NULL,
  `最高` float NOT NULL,
  `最低` float NOT NULL,
  `成交张数` float NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- 資料表新增資料前，先清除舊資料 `stock`
--

TRUNCATE TABLE `stock`;
--
-- 傾印資料表的資料 `stock`
--

INSERT INTO `stock` (`股票代号/名称`, `成交价`, `涨跌`, `涨跌幅`, `最高`, `最低`, `成交张数`) VALUES
('群创', 14.1, 0.7, -4.73, 14.8, 13.9, 473349),
('长荣', 40.7, 2.5, 6.54, 40.7, 38.1, 446253),
('华邦电', 29.05, 0.85, -2.84, 31.15, 28.75, 438038),
('联电', 47.15, 1.15, -2.38, 48.85, 45.9, 302076),
('友达', 14, 0.65, -4.44, 14.7, 13.9, 282578);

-- --------------------------------------------------------

--
-- 資料表結構 `student`
--

DROP TABLE IF EXISTS `student`;
CREATE TABLE `student` (
  `学生姓名` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `国文成绩` float NOT NULL,
  `英文成绩` float NOT NULL,
  `数学成绩` float NOT NULL,
  `班上排名` float NOT NULL,
  `身高` float NOT NULL,
  `体重` float NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- 資料表新增資料前，先清除舊資料 `student`
--

TRUNCATE TABLE `student`;
--
-- 傾印資料表的資料 `student`
--

INSERT INTO `student` (`学生姓名`, `国文成绩`, `英文成绩`, `数学成绩`, `班上排名`, `身高`, `体重`) VALUES
('刘大名', 98, 70, 66, 5, 150, 40),
('John', 50, 50, 50, 17, 177, 80),
('林光英', 77, 40, 90, 15, 160, 53),
('林子幸', 55, 60, 70, 23, 173, 70);

-- --------------------------------------------------------

--
-- 資料表結構 `theater`
--

DROP TABLE IF EXISTS `theater`;
CREATE TABLE `theater` (
  `序号` float DEFAULT NULL,
  `院线公司` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `2018年票房（亿）` float DEFAULT NULL,
  `相关上市公司` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- 資料表新增資料前，先清除舊資料 `theater`
--

TRUNCATE TABLE `theater`;
--
-- 傾印資料表的資料 `theater`
--

INSERT INTO `theater` (`序号`, `院线公司`, `2018年票房（亿）`, `相关上市公司`) VALUES
(21, '武汉天河影业有限公司', 5.89, '/'),
(22, '深圳市深影橙天院线有限公司', 4.79, '/'),
(23, '湖南潇湘影视传播有限责仼公司', 4.64, '/'),
(24, '上海大光明院线有限公司', 4.61, '/'),
(25, '北京长城沃美电影院线有限公司', 4.44, '/');
COMMIT;