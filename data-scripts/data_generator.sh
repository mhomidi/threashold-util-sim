
curr=$PWD/../
cd $2
./script.sh $3
cd $curr
echo $PWD


for i in $(seq 0 1 9)
do
	python $1
	cp *.csv $2/test/$3-$i
	echo "----------$i done---------"	
done
