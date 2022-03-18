import sys
import json
from collections import Counter
import time
import pymongo
	
def inter(a,b):
    return list(set(a)&set(b))
		
def same_path(path0,path):
	if inter(path0["top"],list(path["top"]))!=[] and (((inter(path0["end1"],list(path["end1"]))!=[]) and (inter(path0["end2"],list(path["end2"]))!=[])) or ((inter(path0["end1"],path["end2"])!=[]) and (inter(path0["end2"],path["end1"])!=[]))):
		return True
	else:
		return False

def l1_norm(v):
	s=0
	for i in v:
		s+=abs(i)
	return s

def l1_sim(v1,v2):
	d=[]
	for i in range(len(v1)):
		d.append(v1[i]-v2[i])
	return 1-l1_norm(d)/(l1_norm(v1)+l1_norm(v2))
	
def textsim(token_list,pathtokenlist,add_len,weight):
	ST=0
	for tl in token_list:
		p=t1[4]
		word_list=[]
		v1=[]
		v2=[]
		v3=[]
		for w in tl[0]:
			if (w in word_list) == False:
				word_list.append(w)
				v1.append(1)
				v2.append(0)
				v3.append(weight[add_len+pathtokenlist.index(p+" "+w)])
			else:
				i=word_list.index(w)
				v1[i]+=1
		for j in range(len(tl[1])):
			w=t1[1][j]
			if (w in word_list) == False:
				word_list.append(w)
				v1.append(0)
				v2.append(t1[2][j])
				v3.append(weight[add_len+pathtokenlist.index(p+" "+w)])
			else:
				i=word_list.index(w)
				v2[i]+=t1[2][j]	
		for i in range(len(v1)):
			v1[i]=v1[i]*v3[i]
			v2[i]=v2[i]*v3[i]
		ST_path=l1_sim(v1,v2)
		ST+=(tl[3]*ST_path)
	return ST
	
def Jsimilarity(path0,pathnum0,pathtk0,record,weight,pathtypelist):
	n=0
	d=0
	max=10
	token_list=[]
	path=list(record.keys())
	for i in range(len(record)):
		k=path[i]
		weight_k=weight[pathtypelist.index(k)]
		same=0
		for j in range(len(path0)):
			if path0[j]==k and abs(pathnum0[j]-record[k][0])<max:
				same=1
				w=(max-abs(pathnum0[j]-record[k][0]))/max
				n+=(w*weight_k)
				d+=(2-w)
				token_list.append([pathtk0[j],record[k][1],record[k][2],w,k])
				del path0[j]
				del pathnum0[j]
				del pathtk0[j]
				break
		if same==0:
			d+=weight_k
	for x in path0:
		weight_x=weight[pathtypelist.index(x)]
		d+=weight_x
	jsim=n/d
	for t in token_list:
		t[2]/=n
	return jsim, token_list
	
def count_zero(weight):
	ct=0
	for i in weight:
		if i != 0:
			ct+=1
	return ct
	
def init_weight(v1,v2):
	v_l=len(v1)
	weight=[]
	ct=0
	for i in range(v_l):
		if v1[i]!=0 or v2[i]!=0:
			weight.append(1)
			ct+=1
		else:
			weight.append(0)
	init_w= 1/ct
	for i in range(v_l):
		if weight[i] ==1:
			weight[i]=init_w
	return weight, ct
	
def update_weight(v1, v2, w, total):
	v_l=len(v1)
	ct1=0
	change=[]
	for i in range(v_l):
		if v1[i]==v2[i]:
			if w[i]!=0:
				ct+=1
				w[i]=1
		elif v1[i]==0:
			ct+=1
			w[i]=1
			total+=1
			change.append([i,v2[i]])
		elif v2[i]>v1[i]:
			w[i]=2
		elif v2[i]<v1[i]:
			w[i]=-1
	w_b=1/total
	delta=min(w_b,1-w_b)/2
	w_sum=1
	for i in range(v_l):
		if w[i]==-1:
			w[i]=w_b-delta
			w_sum-=w[i]
		elif w[i] == 2:
			w[i]=w_b+delta
			w_sum-=w[i]
	for i in range(v_l):
		if w[i]==1:
			w[i]=w_sum/ct
	return weight,change
			
				
if __name__ == '__main__':
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	db=myclient["codetrans"]
	tb=db[sys.argv[1]]
	tb_tl=tb[sys.argv[2]]
	
	f1=open("./node/path.json","r")
	f2=open("./node/pathnmn.json","r")
	f3=open("./node/pathtokenn.json","r")
	f4=open("./node/vector.json","r")
	
	path_list0=json.load(f1)
	pathnum_list0=json.load(f2)
	pathtoken_list0=eval(f3.read())
	feature_vector=json.load(f4)
	
	f1.close()
	f2.close()
	f3.close()
	f4.close()
	
	
	if f5=open("./node/vector_fb.json","r"):
		vector_fb=json.load(f5)
		f5.close()
	
	f1=open(t_lang+"_"+s_lang+"_pt.json","r")
	pathtypelist=json.load(f1)
	f1.close()

	f2=open(t_lang+"_"+s_lang+"_tk.json","r")
	pathtokenlist=json.load(f2)
	f2.close()
	
	path_idx=json.load(open("./pbi_index/pbi"+t_lang+"_"+s_lang+".json","w"))
	
	pathtype_list=json.load(open("./pathtype/"+t_lang+"_"+s_lang+".json","r"))
	all_path=pathtype_list["path"]
	for i in range(len(all_path)):
		for j in range(len(path_list0)):
			if same_path(path_list0[j],all_path[i]):
				path_list0[j]=pathtype_list["name"][i]			
	
	tmp=db["temp"]
	for x in tb_tl.find():
		x.pop("_id")
		tmp.insert_one(x)
	for i in range(len(path_list0)):	
		pt=path_list0[i]
		pt_num=pathnum_list0[i]
		pbi=path_idx[pt]
		for j in range(len(pbi)):
			if pt_num < pbi[j]:
				pt_idx= j-1
				break
		tmp1=db["temp1"]
		for x in tmp.find({pt: pt_idx}):
			x.pop("_id")
			tmp1.insert_one(x)
		tmp.drop()
		tmp=tmp1
		tmp1.drop()
		
	for prog in tmp.find():
		record=prog["feature"]
		f_vec=prog["feature_vector"]
		weight, total=init_weight(feature_vector,f_vec)
		add_len=len(pathtypelist)
		if "vector_fb" in dir():
			weight,change=update_weight(feature_vector,vector_fb,weight,total)
			if change != []:
				for c in change:
					if c[0] < add_len:
						ele=pathtypelist[c[0]]
						path_list0.append(ele)
						pathnum_list0.append(c[1])
						pathtoken_list0.append([])
					else:
						ele=pathtokenlist[c[0]-add_len]
						ele_p=ele.split(" ")[0]
						ele_t=ele.split(" ")[1]
						ele_p_idx=path_list0.index(ele_p)
						for i in range(c[i]):
							pathtoken_list0[ele_p_idx].append(ele_t)				
		jsim, token_list=Jsimilarity(path_list0,pathnum_list0,pathtoken_list0,record,weight,pathtypelist)
		ST=textsim(token_list,pathtokenlist,add_len,weight)
		if jsim>0.08:
			finalsim=jsim*0.75+ST*0.25
		else:
			finalsim=0
		
		f5=open("./max.txt","r")
		candidate=f5.readlines()
		if candidate != []:
			if finalsim > float(candidate[1]):
				f5.close()
				f5=open("./max.txt","w")
				f5.write(prog["file"]+"\n"+str(finalsim))
		else:
			f5.close()
			f5=open("./max.txt","w")
			f5.write(prog["file"]+"\n"+str(finalsim))
		f5.close()
