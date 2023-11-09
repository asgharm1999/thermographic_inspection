import tkinter as tk
from preprocessFuncs import preprocess


# Create wrapper function
def preprocessWrapper():
    coldPath = coldPathEntry.get()
    hotPath = hotPathEntry.get()
    savePath = savePathEntry.get()
    method = methodEntry.get()

    preprocess(coldPath, hotPath, savePath, method=method)


# Create window
window = tk.Tk()
window.title("Preprocess")

# Create labels and entry fields
coldPathLabel = tk.Label(window, text="Cold video path")
coldPathLabel.pack()
coldPathEntry = tk.Entry(window)
coldPathEntry.pack()

hotPathLabel = tk.Label(window, text="Hot video path")
hotPathLabel.pack()
hotPathEntry = tk.Entry(window)
hotPathEntry.pack()

savePathLabel = tk.Label(window, text="Save path")
savePathLabel.pack()
savePathEntry = tk.Entry(window)
savePathEntry.pack()

methodLabel = tk.Label(window, text="Method")
methodLabel.pack()
methodEntry = tk.Entry(window)
methodEntry.pack()

# Create button
button = tk.Button(window, text="Button text", command=preprocessWrapper)
button.pack()

# Run window
window.mainloop()



# preprocess(
#     "videos/2023-09-12-15-before-left-straight.mp4",
#     "videos/2023-09-12-15-after-left-straight.mp4",
#     "images/2023-09-12-15-left-straight",
#     method="PCT",
# )
