from PIL import Image
im = Image.open('ejemplo1.webp')
pixelMap = im.load()

im.show()

img = Image.new( im.mode, im.size)

print(pixelMap[0,0])
initial = (0,0,0)
count = 0
for i in range(img.size[0]):
	for j in range(img.size[1]):
		initial = [x+y for x,y in zip(initial, pixelMap[i,j])]
		count+=1
print(initial)
average = [x // count for x in initial]
averageBright = sum(average)//3
average.append(averageBright)
print(average)
print(averageBright)
average = tuple(average)

pixelsNew = img.load()
for i in range(img.size[0]):
	for j in range(img.size[1]):
		if sum(pixelMap[i,j])//3 > averageBright:
			pixelMap[i,j] = average
		else:
			pixelsNew[i,j] = pixelMap[i,j]
img.show()