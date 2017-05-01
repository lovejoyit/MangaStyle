<h2 id="topic">MangaStyle</h2>
This is extended project from my master thesis "Manga-specific Features and Latent Style Model for Manga Style Analysis"
  
By using cnn model I try to classify different manga genre(style).
  
  
<h2 id="dataset">Dataset</h2>
 Use the managa dataset(Manga109) develop by the Aizawa Yamasaki Laboratory, Department of Information and 
 Communication Engineering, the Graduate School of Information Science and Technology, the University of Tokyo. 
 
 download: http://www.manga109.org
   

 <h2 id="code">Code</h2>
 manga_inputdata.py: generate tfrecord file and shuffle the dataset to training set and  testing set.  
 manga_model.py: cnn model.  
 manga_train: training and testing.  
 
  <h2 id="detail">Detail</h2>
In Manga109 dataset, there are 4 kinds of genre, "boy's comic, girl's comic lady's  
comic, young men's comic" I remove some color comic pages, front covers, back  
covers, some pages which are only white or black and painter comment pages.  
Because these pages is not related to the manga drawing style, and divided manga  
 109 dataset to 4 sub dataset based on genre.
  
  

  