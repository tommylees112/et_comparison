// EGU_modis16.js
// Script run on EarthEngine Online code browser

// import library: https://github.com/fitoprincipe/geetools-code-editor/wiki/Batch
var tools = require('users/fitoprincipe/geetools:batch');

// =============================================================================
// read in the MODIS 16 data
// =============================================================================
var modis = ee.ImageCollection('MODIS/006/MOD16A2').filterDate('2001-01-01','2016-01-01');
// var modis = modis.select(['ET','PET','ET_QC']);
var modis = modis.select(['ET']);

// print(modis);

// =============================================================================
// clip to East Africa ROI
// =============================================================================
var bbox=[[32.6, -5.0],[51.8,-5.0],[51.8,15.2],[32.6,15.2]]

var ea_geom = ee.Geometry.Polygon([
      bbox
    ])

// print(ea_geom)
// Map.addLayer(ea_geom)

// filterBounds (WHY DIFFER FROM CLIP?)
var modis = modis.filterBounds(bbox)
print(modis)

var modis_clipped = modis.map(function(image){return image.clip(ea_geom)}) ;
// print(modis_clipped)

Map.addLayer(modis_clipped)

// =============================================================================
// Extract time information from ImageCollection
// =============================================================================

// stack an ImageCollection into image bands
var stackCollection = function(collection) {
    var first = ee.Image(collection.first()).select([]);
    var appendBands = function(image, previous)
    {return ee.Image(previous).addBands(image);};
    return ee.Image(collection.iterate(appendBands, first));
};


// rename the bands by their date
var getScene = function(scene) {
    var dateString = ee.Date(scene.get('system:time_start')).format('yyyy-MM-dd');
    // var mask = scene.select('NDVI').gt(-9999).updateMask(ee.Image(1));
    return scene.rename(dateString);
};


var stackColl = stackCollection(modis_clipped.map(getScene));
print(stackColl);

// generate a datelist
var datelist = ee.List(stackColl.bandNames()).map(function(date){return ee.Date(date).format('yyyy-MM-dd')});

print(datelist)

// var et_stacked = stackCollection(modis_clipped.rename(datelist));
// print(et_clipped)
// =============================================================================
// Extract time information from ImageCollection
// =============================================================================

// var count = modis_clipped.size();
//
// //List ET
// var list=modis_clipped.toList(count);
//
// // for each image in the imageCollection (TIME)
// for (var i=0; i<count; i++){
//          var image = ee.Image(list.get(i));
//          var date = image.date().format('yyyy-MM-dd').getInfo()
//          var name = 'ET_' + i.toString() + '_' + date
//          print(name);
//      // Export ET
//      var task = Export.image.toDrive({
//                    image: image,
//                    description: name,
//                    fileNamePrefix: name,
//                    scale: 500,
//                    region:ea_geom,
//                    crs : 'EPSG:4326'
//                });
//     task.start()
//     }

// =============================================================================
// DOWNLOAD
// =============================================================================
var scale  = 500;
var crs='EPSG:4326';

// RUN THE TASK
// var task = Export.image.toDrive({
//   image: stackColl,
//   description: 'modis_et',
//   scale: scale,
//   region: ea_geom,
//   folder: 'modis'
// })

// https://gis.stackexchange.com/questions/248216/export-each-image-from-a-collection-in-google-earth-engine
// tools.Download.ImageCollection.toDrive(modis_clipped,'modis',
//     {scale:scale,
//     region:ea_geom,
//     type:'float',
//     }
// )



//
// // Append all of the bands into ONE image
// var appendBand = function(current, previous){
//     // # Rename the band
//     previous=ee.Image(previous);
//     current = current.select([0]);
//     // # Append it to the result (Note: only return current item on first element/iteration)
//     var accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous, null), current, previous.addBands(ee.Image(current)));
//     // # Return the accumulation
//     return accum;
// }
// var img = modis_clipped.iterate(appendBand)
// print(img)



// // RUN THE TASK
// var task = Export.image.toDrive({
//   image: modis,
//   description: 'modis_et',
//   scale: scale,
//   region: ea_geom,
//   folder: 'modis'
// })
