# make a file with each row having one number: 0, 1, 2, 3, .... for 200 independent runs of the same script on slurm

with open('run_list.txt','w') as fil:
     for i in range(200):
         fil.write(str(i)+'\n')
