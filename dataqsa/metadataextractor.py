import sys
import exifread


f = open('../sample/DSC_V1_6460_2238.JPG', 'rb')
tags = exifread.process_file(f)

#Show all Keys
for tmp in tags.keys():
    print(tmp, tags[tmp])


meta_aufnahme_datum =  tags['EXIF DateTimeOriginal'].values
meta_aufnahme_width = tags['EXIF ExifImageWidth'].values
meta_aufnahme_height  = tags['EXIF ExifImageLength'].values
meta_aufnahme_iso = tags['EXIF ISOSpeedRatings'].values

print(meta_aufnahme_datum)
print(meta_aufnahme_width)
print(meta_aufnahme_height)
print(meta_aufnahme_iso)

