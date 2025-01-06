# FastICA Demo Code

source data: http://www.openslr.org/12 test-clean dataset
speaker 1: /test-clean/237/134500/237-134500-0031.flac
speaker 2: /test-clean/908/31957/908-31957-0008.flac

converted flac to mp3 with: ffmpeg -i [input].flac -ab 320k -map_metadata 0 -id3v2_version 3 [output].mp3