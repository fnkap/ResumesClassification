# for i in range(50):
#     with open ("{}.txt".format(i+1), "w+") as file:
#         pass

import pyperclip

i = 1
my_clip = pyperclip.paste()
old_clip = pyperclip.paste()
while True:
    my_clip = pyperclip.paste()
    if my_clip != old_clip and my_clip is not None:
        print("Writing {}.txt".format(i))
        try:
            with open("{}.txt".format(i), "w+", encoding="utf-8") as file:
                file.write(my_clip)
            i += 1
            old_clip = my_clip
        except Exception as e:
            print("Errore sul {}. Lo salto.".format(i))
            pass
