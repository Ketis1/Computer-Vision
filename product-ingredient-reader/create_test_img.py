from PIL import Image, ImageDraw, ImageFont

img = Image.new('RGB', (400, 200), color=(255, 255, 255))
d = ImageDraw.Draw(img)

text = "Ingredients: Water, Sugar, Carbon Dioxide\nNutrition Facts: 100 kcal"
d.text((20, 50), text, fill=(0, 0, 0))

img.save('test_label.png')
print("test_label.png created!")
