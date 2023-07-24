
curr=$PWD
cd $2
./script.sh $3
cd $curr
echo $PWD


for i in $(seq 0 1 9)
do
	python $1 5
	cp *.csv $2/$3-$i
	echo "----------$i done---------"	
done
