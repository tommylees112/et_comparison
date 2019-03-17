// https://gis.stackexchange.com/questions/279964/export-images-in-google-earth-engine-with-date-name

// Load a FeatureCollection
var violada = ee.FeatureCollection("ft:1KmH70D7VKdiWocelf0RtbL_kQhGk0LGkQQ7O-ceG");

// Load a FeatureCollection Sentinel 2.
var s2 = ee.ImageCollection('COPERNICUS/S2')
          .filterBounds(violada)
          .filterDate('2016-10-01', '2017-09-30');
print(s2);

// Function to mask clouds using the Sentinel-2 QA band..
function maskS2clouds(image) {
  // get date of the image to pass it through
  var date = image.date().millis()
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = ee.Number(2).pow(10).int();
  var cirrusBitMask = ee.Number(2).pow(11).int();

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(
             qa.bitwiseAnd(cirrusBitMask).eq(0));

  // Return the masked and scaled data.
  return image.updateMask(mask).divide(10000).set('system:time_start', date);
}
var imagen=s2.map(maskS2clouds);

print(ee.Image(imagen.first()))

// visualize the first image in the collection, pre- and post- mask
var visParams = {bands: ['B8','B4','B3']}
Map.addLayer(ee.Image(imagen.first()), visParams, 'clouds masked', false)
Map.addLayer(ee.Image(s2.first()), visParams, 'original', false)

//add ndwi
var addNDVI = function(image) {
  return image.addBands(image.normalizedDifference(['B8','B4']).rename('NDVI'));
};
var imagen2=imagen.map(addNDVI);

var NDVI = imagen2//.select(['NDVI']);
print(ee.Image(NDVI.first()));

var composite = NDVI.qualityMosaic('NDVI').clip(violada);
print(composite);

// Visualize NDVI
var ndviPalette = ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718',
  '74A901', '66A000', '529400', '3E8601', '207401', '056201',
  '004C00', '023B01', '012E01', '011D01', '011301'];
Map.addLayer(composite.select('NDVI'),
            {min:0, max: 1, palette: ndviPalette}, 'ndvi');


//List NDVI
var list=NDVI.toList(150);

// for each image in the imageCollection (TIME)
for (var i=0;i<150;i++){
         var image = ee.Image(list.get(i));
         var date = image.date().format('yyyy-MM-dd').getInfo()
         var name= 'NDVI_'+i.toString()+'_'+date
         print(name);
    // Export NDVI
         Export.image.toDrive({
               image: image,
               description: name,
               fileNamePrefix: name,
               scale: 10,
               region:violada,
               crs : 'EPSG:32630'
               });
    }
Map.setCenter(-0.6, 42, 12);
