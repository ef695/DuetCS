source ~/.bash_profile
if [ -f "./max.txt" ];then
	rm max.txt
fi
touch max.txt

fb=0
files=$(ls "./")
for file in $files
do
    if [[ $file =~ "feedback" ]]
    then
        fb=$file
    fi
done

if [ $fb =0 ];then
	file=$1
	#generate CST for input
	if [ "${file##*.}"x = "js"x ];then
		grun JavaScript program -tree $file > test.txt
		s="JavaScript"
	elif [ "${file##*.}"x = "java"x ];then
		grun Java8 compilationUnit -tree $file > test.txt
		s="Java8"
	elif [ "${file##*.}"x = "py"x ];then
		grun Python3 file_input -tree $file > test.txt
		s="Python3"
	elif [ "${file##*.}"x = "cpp"x ];then
		grun CPP14 translationunit -tree $file > test.txt
		s="CPP14"
	fi
	t="./test.txt"
	python3 proc_input.py $s $2 $t
else
	file=$fb
	#generate CST for input
	if [ "${file##*.}"x = "js"x ];then
		grun JavaScript program -tree $file > test_fb.txt
		s="JavaScript"
	elif [ "${file##*.}"x = "java"x ];then
		grun Java8 compilationUnit -tree $file > test_fb.txt
		s="Java8"
	elif [ "${file##*.}"x = "py"x ];then
		grun Python3 file_input -tree $file > test_fb.txt
		s="Python3"
	elif [ "${file##*.}"x = "cpp"x ];then
		grun CPP14 translationunit -tree $file > test_fb.txt
		s="CPP14"
	fi
	t="./test_fb.txt"
	python3 proc_input.py $s $2 $t
	rm $t
fi
python3 retrieval_fb.py $s $2
python3 results.py
