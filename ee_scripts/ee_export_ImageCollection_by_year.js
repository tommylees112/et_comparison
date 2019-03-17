// https://gis.stackexchange.com/questions/287190/imagecollection-exporting-by-years

var col = ee.ImageCollection("MODIS/006/MOD13Q1")
    .select("NDVI")
    .filterDate("2000-01-01","2018-06-01");

// Functions to stack colletion series into image bands

var stackCollection = function(collection) {
  var first = ee.Image(collection.first()).select([]);
  var appendBands = function(image, previous)
  {return ee.Image(previous).addBands(image);};
  return ee.Image(collection.iterate(appendBands, first));};

// Function to generate mask for each Scene.

var getSceneMask = function(scene) {
  var dateString = ee.Date(scene.get('system:time_start')).format('yyyy-MM-dd');
  var mask = scene.select('NDVI').gt(-9999).updateMask(ee.Image(1));
  return mask.rename(dateString);};

//Stacked mask for collection

var MaskCol = stackCollection(col.map(getSceneMask));

// Generate a datelist for collection

var datelist = ee.List(MaskCol.bandNames()).map(function(date){return ee.Date(date).format('yyyy-MM-dd')});

print(datelist)

// Stack bands and rename by date

var ndvi_stacked = stackCollection(col.select('NDVI')).divide(10000).rename(datelist);

Map.addLayer(ndvi_stacked,{min:-1,max:1},'ndvi_stacked');
