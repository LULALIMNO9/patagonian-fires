// ============================================================
// Burned area and burn severity mapping with Sentinel-2
// Methods: NBR, dNBR, NDVI, dNDVI, water and shoreline exclusion
// Author: Lucía G. De Stefano
// ============================================================
//
// Requirements
//   ROI       Polygon (Geometry or FeatureCollection) drawn or imported in the
//             Code Editor.
//   TRAINING  (optional) FeatureCollection of labelled polygons with a numeric
//             property 'class' (1 = burned, 0 = not burned). When present, the
//             script also exports pixel samples for the Python classifier.
//
// Exports (Google Drive, folder params.folder)
//   <prefix>_features        8-band float stack (indices pre/post and differences)
//   <prefix>_severity        dNBR severity classes (0-4, 255 = excluded)
//   <prefix>_candidate_burn  binary map of candidate burned area
//   <prefix>_training_samples  CSV with features at TRAINING pixels (optional)

// 1. PARAMETERS ------------------------------------------------------

var params = {
  // Pre-fire window must end before the fire starts; post-fire window after it is contained.
  startPre: '2025-01-01',
  endPre: '2025-01-29',
  startPost: '2025-02-15',
  endPost: '2025-03-15',

  cloudLimit: 80,          // max scene cloud percentage (pixels are masked anyway)
  scale: 20,               // m

  mndwiWater: 0.10,        // MNDWI above this value = water
  ndviMin: 0.20,           // pre-fire NDVI above this value = vegetated
  dnbrMin: 0.10,           // candidate burned: dNBR above ...
  dndviMin: 0.10,          // ... and dNDVI above ...
  shoreBufferM: 40,        // pixels within this distance of water are excluded
  minPatchPixels: 0,       // remove candidate patches smaller than this (0 = off)

  // dNBR severity class limits (Key & Benson, 2006)
  severityLimits: [0.10, 0.27, 0.44, 0.66],

  nodata: -9999,
  folder: 'GEE_burn_severity',
  prefix: 'burn'
};

var region = ee.FeatureCollection(ROI).geometry();


// 2. SENTINEL-2 COMPOSITES AND CLOUD MASK ----------------------------

// Scene classification (SCL) values removed: saturated/defective (1), cloud
// shadow (3), unclassified (7), cloud medium/high probability (8, 9),
// thin cirrus (10), snow/ice (11).
function maskClouds(image) {
  var scl = image.select('SCL');
  var mask = scl.neq(1).and(scl.neq(3)).and(scl.neq(7))
    .and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));

  var reflectance = image.select(['B2', 'B3', 'B4', 'B8', 'B8A', 'B11', 'B12'])
    .multiply(0.0001);

  return ee.Image(reflectance.addBands(scl)
    .updateMask(mask)
    .copyProperties(image, ['system:time_start']));
}

var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region)
  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', params.cloudLimit));

var rawPre = collection.filterDate(params.startPre, params.endPre);
var rawPost = collection.filterDate(params.startPost, params.endPost);

print('Pre-fire images:', rawPre.size());
print('Post-fire images:', rawPost.size());

var pre = rawPre.map(maskClouds).median().clip(region);
var post = rawPost.map(maskClouds).median().clip(region);

// Most frequent scene class (unmasked) for water detection
var sclPre = rawPre.select('SCL').mode().clip(region);
var sclPost = rawPost.select('SCL').mode().clip(region);


// 3. SPECTRAL INDICES ------------------------------------------------

function indices(image, suffix) {
  return ee.Image.cat([
    image.normalizedDifference(['B8A', 'B12']).rename('NBR_' + suffix),
    image.normalizedDifference(['B8', 'B4']).rename('NDVI_' + suffix),
    image.normalizedDifference(['B3', 'B11']).rename('MNDWI_' + suffix)
  ]);
}

var preIdx = indices(pre, 'pre');
var postIdx = indices(post, 'post');

var dNBR = preIdx.select('NBR_pre').subtract(postIdx.select('NBR_post')).rename('dNBR');
var dNDVI = preIdx.select('NDVI_pre').subtract(postIdx.select('NDVI_post')).rename('dNDVI');

var features = ee.Image.cat([
  preIdx.select('NBR_pre'), postIdx.select('NBR_post'), dNBR,
  preIdx.select('NDVI_pre'), postIdx.select('NDVI_post'), dNDVI,
  preIdx.select('MNDWI_pre'), postIdx.select('MNDWI_post')
]).toFloat();


// 4. WATER, SHORELINE AND VEGETATION MASKS ---------------------------

var waterPre = sclPre.eq(6).or(preIdx.select('MNDWI_pre').gt(params.mndwiWater));
var waterPost = sclPost.eq(6).or(postIdx.select('MNDWI_post').gt(params.mndwiWater));

var waterBuffer = waterPre.or(waterPost).focalMax({
  radius: params.shoreBufferM,
  units: 'meters'
});
var landMask = waterBuffer.not();

var vegetationMask = preIdx.select('NDVI_pre').gt(params.ndviMin)
  .and(preIdx.select('MNDWI_pre').lt(params.mndwiWater));

var analysisMask = landMask.and(vegetationMask);


// 5. CANDIDATE BURNED AREA -------------------------------------------

var candidateBurn = dNBR.gt(params.dnbrMin)
  .and(dNDVI.gt(params.dndviMin))
  .and(analysisMask)
  .selfMask()
  .rename('candidate_burn');

if (params.minPatchPixels > 0) {
  var patchSize = candidateBurn.connectedPixelCount(params.minPatchPixels + 1, true);
  candidateBurn = candidateBurn.updateMask(patchSize.gte(params.minPatchPixels));
}


// 6. BURN SEVERITY CLASSES -------------------------------------------
// 0 unburned, 1 low, 2 moderate-low, 3 moderate-high, 4 high

var severity = ee.Image(0);
params.severityLimits.forEach(function (limit) {
  severity = severity.add(dNBR.gte(limit));
});
severity = severity.updateMask(analysisMask).rename('severity').toByte();


// 7. AREA STATISTICS (ha) --------------------------------------------

var pixelHa = ee.Image.pixelArea().divide(10000);

var candidateHa = pixelHa.updateMask(candidateBurn).reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: region,
  scale: params.scale,
  maxPixels: 1e13,
  tileScale: 4
});
print('Candidate burned area (ha):', candidateHa.get('area'));

var severityHa = pixelHa.addBands(severity).reduceRegion({
  reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'severity_class'}),
  geometry: region,
  scale: params.scale,
  maxPixels: 1e13,
  tileScale: 4
});
print('Area by severity class (ha):', severityHa.get('groups'));


// 8. MAP ------------------------------------------------------------

Map.centerObject(region, 10);

var rgbVis = {bands: ['B4', 'B3', 'B2'], min: 0.02, max: 0.30};
Map.addLayer(pre, rgbVis, 'RGB pre-fire', false);
Map.addLayer(post, rgbVis, 'RGB post-fire', false);
Map.addLayer(post, {bands: ['B12', 'B8A', 'B4'], min: 0.02, max: 0.40}, 'SWIR post-fire', false);
Map.addLayer(waterBuffer.selfMask(), {palette: ['#00BFFF']}, 'Water and shoreline (excluded)', false);
Map.addLayer(dNBR.updateMask(landMask), {
  min: -0.3, max: 0.8,
  palette: ['#2166ac', '#f7f7f7', '#fee08b', '#f46d43', '#a50026']
}, 'dNBR (land)', true);
Map.addLayer(severity, {
  min: 0, max: 4,
  palette: ['#d9f0d3', '#fee08b', '#fdae61', '#f46d43', '#a50026']
}, 'Burn severity', false);
Map.addLayer(candidateBurn, {palette: ['#ff0000']}, 'Candidate burned area', true);


// 9. EXPORTS --------------------------------------------------------

Export.image.toDrive({
  image: features.updateMask(landMask).unmask(params.nodata),
  description: params.prefix + '_features',
  folder: params.folder,
  region: region,
  scale: params.scale,
  maxPixels: 1e13
});

Export.image.toDrive({
  image: severity.unmask(255).toByte(),
  description: params.prefix + '_severity',
  folder: params.folder,
  region: region,
  scale: params.scale,
  maxPixels: 1e13
});

Export.image.toDrive({
  image: candidateBurn.unmask(0).toByte(),
  description: params.prefix + '_candidate_burn',
  folder: params.folder,
  region: region,
  scale: params.scale,
  maxPixels: 1e13
});


// 10. TRAINING SAMPLES (optional) -----------------------------------

if (typeof TRAINING !== 'undefined') {
  var polygons = ee.FeatureCollection(TRAINING).map(function (f) {
    return f.set('poly_id', f.id());
  });

  var samples = features.updateMask(landMask).sampleRegions({
    collection: polygons,
    properties: ['class', 'poly_id'],
    scale: params.scale,
    geometries: true,
    tileScale: 4
  });

  print('Training samples:', samples.size());

  Export.table.toDrive({
    collection: samples,
    description: params.prefix + '_training_samples',
    folder: params.folder,
    fileFormat: 'CSV'
  });
}
