from sklearn.feature_extraction.text import CountVectorizer
import codecs
import sys
import scipy.io

reload(sys)
sys.setdefaultencoding('utf8')

with codecs.open(
        '/Users/Vishal/Desktop/CS Classes/CS467/GameOfDebates/DebateAnalysis/transcripts/Republicans/rep_1-14-2016-wl.txt'
        , 'r', encoding='utf-8', errors='ignore') as debate:
    debate_text = debate.read().strip(' \t\n\r')

debate_text = debate_text.splitlines()
print debate_text[0]

vectorizer = CountVectorizer(min_df=1)

X = vectorizer.fit_transform(debate_text)
Y = vectorizer.get_feature_names()

print X
print Y

vocab_y = open('/Users/Vishal/Desktop/CS Classes/CS467/GameOfDebates/DebateAnalysis/Dataset/vocab.debate.txt', 'w')
for word in Y:
    vocab_y.write("%s\n" % word)


vocab_x = open('/Users/Vishal/Desktop/CS Classes/CS467/GameOfDebates/DebateAnalysis/Dataset/docword.debate.txt', 'w')
scipy.io.mmwrite(vocab_x, X)
