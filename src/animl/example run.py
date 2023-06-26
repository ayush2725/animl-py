from plotboxes import drawBoundingBoxes

data = pd.read_csv("fake_data.csv")

# Test with the first row
drawBoundingBoxes(data.iloc[0:1], 65, "/home/srinidhiyerbati/animl-py/animl-py/src/animl/")

# Test with all rows
for i, image in data.iterrows():
    drawBoundingBoxes(image, 60 + i, "/home/srinidhiyerbati/animl-py/animl-py/src/animl/"
