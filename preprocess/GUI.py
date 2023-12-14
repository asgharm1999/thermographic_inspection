import tkinter as tk
from tkinter import filedialog
from preprocessFuncs import preprocess
from PIL import Image, ImageTk


# Create button functions
def coldButtonFunc():
    coldPath = filedialog.askopenfilename(title="Select cold video")
    if coldPath:
        coldPathEntry.delete(0, tk.END)
        coldPathEntry.insert(0, coldPath)


def hotButtonFunc():
    hotPath = filedialog.askopenfilename(title="Select hot video")
    if hotPath:
        hotPathEntry.delete(0, tk.END)
        hotPathEntry.insert(0, hotPath)


# Create wrapper function
def preprocessWrapper():
    coldPath = coldPathEntry.get()
    hotPath = hotPathEntry.get()
    savePath = savePathEntry.get()
    method = methodChosen.get()

    if not coldPath:
        # Display error message in window
        coldPathLabel.config(text="No cold video path selected")
        return
    if not hotPath:
        hotPathLabel.config(text="No hot video path selected")
        return
    if not savePath:
        savePathLabel.config(text="No save path selected")
        return
    if not method:
        methodLabel.config(text="No method selected")
        return

    figurePath = preprocess(coldPath, hotPath, savePath, method=method)
    image = Image.open(figurePath)
    photo = ImageTk.PhotoImage(image)
    imageLabel.config(image=photo)
    imageLabel.image = photo


# Create window
window = tk.Tk()
window.title("Preprocess")

# Create labels and entry fields
coldPathLabel = tk.Label(window, text="Cold video path")
coldPathLabel.pack()
coldPathEntry = tk.Entry(window, width=40)
coldPathEntry.pack()
coldButton = tk.Button(window, text="Browse", command=coldButtonFunc)
coldButton.pack()

hotPathLabel = tk.Label(window, text="Hot video path")
hotPathLabel.pack()
hotPathEntry = tk.Entry(window, width=40)
hotPathEntry.pack()
hotButton = tk.Button(window, text="Browse", command=hotButtonFunc)
hotButton.pack()

savePathLabel = tk.Label(window, text="Save path")
savePathLabel.pack()
savePathEntry = tk.Entry(window, width=40)
savePathEntry.pack()

methodLabel = tk.Label(window, text="Method")
methodLabel.pack()
methodChosen = tk.StringVar()
methodOptions = ["PCT", "SPCT", "PPT"]
methodDropdown = tk.OptionMenu(window, methodChosen, *methodOptions)
methodDropdown.pack()

imageLabel = tk.Label(window, text="Result will be displayed below")
imageLabel.pack()

# Create button
button = tk.Button(window, text="Begin preprocessing", command=preprocessWrapper)
button.pack()

# Run window
window.mainloop()



# preprocess(
#     "videos/2023-09-12-15-before-left-straight.mp4",
#     "videos/2023-09-12-15-after-left-straight.mp4",
#     "images/2023-09-12-15-left-straight",
#     method="PCT",
# )
