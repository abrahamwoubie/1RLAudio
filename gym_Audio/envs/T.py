from scipy.spatial import distance
dist = {}
state={}
num_rows=4
num_cols=4

for i in range(0,num_rows):
    for j in range(0,num_cols):
        goal=(num_rows-1,num_cols-1)
        current=(i,j)
        dist[i,j] = round(distance.euclidean(goal, current),2)
        print(dist[i,j])

for i in range(0,num_rows):
    for j in range(0,num_cols):
        #print(i,j,'\n')
        if(dist[i,j]!=0):
            if (i < num_rows - 1):
                down = dist[i + 1, j]
            else:
                down = 100

            if (j < num_cols - 1):
                right = dist[i, j + 1]
            else:
                right = 100

            if (down<right):
                i = i + 1
                j=i
                break
                #print("i,j", i, j)
            else:
                j = j + 1
                #i=j
                #print("ii,jj", i, j)
                #break
            print("Current state ",i,j)