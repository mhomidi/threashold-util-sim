

curr=$PWD
cd $curr
mkdir -p $curr/scripts/logs

for i in $(seq 20 1 60)
do
	python test/test.py 1 0 0 0 20 $i | grep "sched time" | grep -oE '[0-9]+\.[0-9]+' >> scripts/logs/mtf-ag.csv
	python test/test.py 1 0 0 0 $i 40 | grep "sched time" | grep -oE '[0-9]+\.[0-9]+' >> scripts/logs/mtf-ac.csv

	python test/test.py 1 0 1 1 20 $i | grep "sched time" | grep -oE '[0-9]+\.[0-9]+' >> scripts/logs/gf-ag.csv
	python test/test.py 1 0 1 1 $i 40 | grep "sched time" | grep -oE '[0-9]+\.[0-9]+' >> scripts/logs/gf-ac.csv

	python test/test.py 1 0 1 2 20 $i | grep "sched time" | grep -oE '[0-9]+\.[0-9]+' >> scripts/logs/rr-ag.csv
	python test/test.py 1 0 1 2 $i 40 | grep "sched time" | grep -oE '[0-9]+\.[0-9]+' >> scripts/logs/rr-ac.csv

    echo "Finish for $i"
done