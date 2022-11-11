import json
import operator

def treeVote(tree,object):
    if '好瓜' == tree or '坏瓜' == tree:
        return tree
    for feature in tree.keys():
        for key in tree.get(feature):
    #用增强for对python字典的get方法获取某一key的value，对get的result进行遍历，如果该result还是个字典的话，
    # 对get的result进行遍历是对result的key进行遍历
    #噢，python中对字典用增强for遍历，获取的是其内容key
            if key in object:
                return treeVote(tree.get(feature).get(key), object)

def voteType(typeList):  # 在叶子节点处（属性节点用完了），如果还是多种类别混杂，则根据哪种类别最多，就判为哪种类别，同理此方法也可以用于最后森林投票处，把所有投票结果放进来，然后获取最大的
    typeCount = {}
    for vote in typeList:
        if vote not in typeCount.keys():
            typeCount[vote] = 0
        typeCount[vote] += 1
    sortedTypeCount = sorted(typeCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedTypeCount[0][0]



if __name__ == '__main__':
    f=open("json/treeset.json", "r")
    treeSet=json.loads(f.read())
    typeList=[]
    object=input()
    for tree in treeSet:
        typeList.append(treeVote(tree,object))
    print(voteType(typeList))