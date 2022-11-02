# Coronal-Mass-Ejection-VR

## ...
## ...

## Sample data 

Current data in mostly-plain-text ".speck" form:
- [folder containing ~50GB, every 10th timestep](https://virdir.ncsa.illinois.edu/stuffed/slevy/VR-CME/current/)

The .speck file format contains plain text, with one 3-D point plus attributes on each line, such as:
   1.55807 -0.00576553 -0.327232 7.65378
The first three give the position in Cartesian coordinates, the fourth number is the current.

Some other lines can appear too:
- comments, beginning with "#"
- "datavar" (specifying the attribute name for the N'th attribute, as in "datavar 0 current")
- "maxcomment N" (meaningful to partiview, but you can ignore it)

In this case the only attribute is the current-density (curl B, with some radial normalization).

Also included in the above folder are some ".pb" files -- it's a simple binary particle file format that I made up.
You can make up your own, or use this one.   See scripts/pbio.py in this repo for a description of the file format.

Here, current/current.NNNN.speck contains the same information as current/current.NNNN.pb.

There is a python reader/writer for the .pb format in scripts/pbio.py.

Pre-visualization using partiview:
- [partiview software (downloads, etc)](https://virdir.ncsa.illinois.edu/partiview/)
- [subsetted data, use as: partiview current+lines0923.cf](https://virdir.ncsa.illinois.edu/stuffed/slevy/VR-CME/partiview-CME-preview.tar.gz)


