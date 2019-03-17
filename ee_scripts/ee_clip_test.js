// GET the imageCollection (MODIS)
var imgcoll = ee.ImageCollection('MODIS/MOD09A1').filterBounds(ee.Geometry.Rectangle(-106.5, 50,-64, 23)).filterDate('2001-12-31','2015-12-31')
var imgcoll2 = ee.ImageCollection('MODIS/051/MCD12Q1').filterBounds(ee.Geometry.Rectangle(-106.5, 50,-64, 23)).filterDate('2001-12-31','2015-12-31')
var temp = ee.ImageCollection('MODIS/MYD11A2').filterBounds(ee.Geometry.Rectangle(-106.5, 50,-64, 23)).filterDate('2001-12-31','2015-12-31')

print(temp)

var lat = -86.646
var lon = 32.532
var offset =0.11

var listOfImages = imgcoll.toList(imgcoll.size());
var listOfImages2 = imgcoll2.toList(imgcoll.size());

// // GET the state borders shapefiles
var county_region = ee.FeatureCollection('ft:18Ayj5e7JxxtTPm1BdMnnzWbZMrxMB49eqGDTsaSp')
var county_region2 = ee.FeatureCollection('ft:1S4EB6319wWW2sWQDPhDvmSBIVrD3iEmCLYB7nMM')

var region = county_region.filterMetadata('STATE num', 'equals', 1)
var region = ee.FeatureCollection(region).filterMetadata('COUNTY num', 'equals', 1)

var img1 = ee.Image(listOfImages.get(0)).clip(region);
var img2 = ee.Image(listOfImages.get(0)).clip(region);

// Map.addLayer(county_region)
// Map.addLayer(county_region2)

var world_region = ee.FeatureCollection('ft:1tdSwUL7MVpOauSgRzqVTOwdfy17KDbw-1d9omPw')

print(world_region)
Map.addLayer(world_region)

// =============================================================================
// APPEND BAND FUNCTION
// =============================================================================
var appendBand = function(current, previous){
    // # Rename the band
    previous=ee.Image(previous);
    current = current.select([0]);
    // # Append it to the result (Note: only return current item on first element/iteration)
    var accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous, null), current, previous.addBands(ee.Image(current)));
    // # Return the accumulation
    return accum;
}
var img=imgcoll.iterate(appendBand)

// =============================================================================
// https://gis.stackexchange.com/questions/265445/export-time-series-modis-16-day-ndvi-and-evi-data-on-google-earth-engine?rq=1
// =============================================================================

var NDVICollection=ee.ImageCollection('MODIS/MOD13Q1')
                     .filterDate('2000-01-01','2017-12-31')
                     .filterBounds(AOI).select('NDVI');

var list=NDVICollection.toList(length);//the length mean that the volume of collection

for (var i=0;i<length;i++){
     var image=ee.Image(list.get(i));
     var time=image.get('system:index');
     var name=ee.String(time);

     Export.image.toDrive({
         image: image,
         description: name,
         scale: 30,
         maxPixels:1e13
});
};




// var region2 = county_region2.filterMetadata('STATE num', 'equals', 1)
// var region2 = ee.FeatureCollection(region2).filterMetadata('COUNTY num', 'equals', 1)

// // NOTE: cannot filterBounds a collection
// // have to apply the function to each variable using map
// var out = imgcoll.map(
//   function(image) {
//     return image.clip(region);

//   })
//   ;

// // SUBSET your image collection using lists!
// var listOfImages = imgcoll.toList(imgcoll.size());
// var outList = out.toList(out.size())

// // var img1 = ee.Image(outList.get(0))
// var img1 = ee.Image(listOfImages.get(0)).clip(region);
// var img2 = ee.Image(listOfImages.get(0)).clip(region2);

// // print to view in the console
// print(img1);
// print(img2);

// print(img1.bandNames());

// var vizParams = {
//   bands: ['sur_refl_b01', 'sur_refl_b01', 'sur_refl_b01'],
//   min: 0,
//   max: 0.5,
//   gamma: [0.95, 1.1, 1]
// };

// // ADD to map to visualise
// Map.addLayer(img1, vizParams, "layerSP");
// Map.addLayer(img2, vizParams, "layerMM");
