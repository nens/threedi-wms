var map = L.map('map');

// Layer definitions
var osmUrl='http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
var osmAttrib='Map data © OpenStreetMap contributors';
var osm = new L.TileLayer(osmUrl, {minZoom: 8, maxZoom: 13, attribution: osmAttrib, zIndex: 0});

var bathymetry = new L.tileLayer.canvas({zIndex: 1});
bathymetry.drawTile = function(canvas, tilePoint, zoom) {
  loadBathymetryTile(canvas, zoom, tilePoint.x, tilePoint.y);
}

var depth = new L.tileLayer.canvas({zIndex: 2});
depth.drawTile = function(canvas, tilePoint, zoom) {
  loadDepthTile(canvas, zoom, tilePoint.x, tilePoint.y);
}

var grid = new L.tileLayer.canvas({zIndex: 3});
grid.drawTile = function(canvas, tilePoint, zoom) {
  loadGridTile(canvas, zoom, tilePoint.x, tilePoint.y);
}
  
// Tile loaders
function loadBathymetryTile(canvas, zoom, x, y) {  
  var url = "/3di/tms/" + zoom + "/" + x + "/" + y + ".png";
  url += "?layer=" + getLayer() + ":bathymetry";
  url += "&limits=" + info['limits'][0] + "," + info['limits'][1];
  url += "&antialias=" + getAntialias();
  canvas.img = new Image();
  canvas.img.src = url
  canvas.img.onload = drawData.bind(canvas);
}
function loadDepthTile(canvas, zoom, x, y) {  
  var url = "/3di/tms/" + zoom + "/" + x + "/" + y + ".png";
  url += "?layer=" + getLayer() + ":depth";
  url += "&time=" + getTime();
  url += "&antialias=" + getAntialias();
  canvas.img = new Image();
  canvas.img.src = url
  canvas.img.onload = drawData.bind(canvas);
}
function loadGridTile(canvas, zoom, x, y) {  
  var url = "/3di/tms/" + zoom + "/" + x + "/" + y + ".png";
  url += "?layer=" + getLayer() + ":grid";
  url += "&antialias=" + getAntialias();
  canvas.img = new Image();
  canvas.img.src = url
  canvas.img.onload = drawData.bind(canvas);
}
function drawData(){
  var ctx = this.getContext('2d');
  ctx.clearRect(0, 0, 256, 256);
  ctx.drawImage(this.img, 0, 0);
}

// Main init function
function updateLayer(){
  // Determine bounds
  $.ajax(
    '/3di/wms',
    { 
      data: {
        request: 'getinfo',
        layer: getLayer(),
        srs: 'epsg:4326'
      },
      success: updateInfo
    }
  );
}
function updateInfo(data){
  // There'
  info = data;
  updateSlider();
  bathymetry.redraw();
  depth.redraw();
  grid.redraw();
  var bounds = info['bounds'];
  map.fitBounds([[bounds[1], bounds[0]],
                 [bounds[3], bounds[2]]])
}

// Dom access functions
function getAntialias(){
  if ($("input#antialias").is(":checked")) {
    return 'yes';
  } else {
    return 'no';
  }
}
function getLayer(){
  return $('select#layer option:selected').val();
}

// Slider
function slide(ui, slider){
  $("#time").text(slider.value);
  depth.redraw();
}
function getTime(){
  return $("#time").text();
}
function setTime(time){
  $('#slider').slider("option", "value", time);
  $("#time").text(time);
}
function updateSlider() {
  var sliderMax = info['timesteps'] - 1;
  $("#slider").slider("option", "max", sliderMax);
  if (getTime() > sliderMax) {setTime(sliderMax)}
}
$("#slider").slider({
  min: 0,
  max: 143,
  slide: slide
});


// Layer toggles
function toggleOsm(){
  var state = $("input#osm").is(":checked");
  if (state == false) {
    map.removeLayer(osm);
  } else {
    map.addLayer(osm);
  }
}
function toggleBathymetry(){
  var state = $("input#bathymetry").is(":checked");
  if (state == false) {
    map.removeLayer(bathymetry);
  } else {
    map.addLayer(bathymetry);
  }
}
function toggleDepth(){
  var state = $("input#depth").is(":checked");
  if (state == false) {
    map.removeLayer(depth);
  } else {
    map.addLayer(depth);
  }
}
function toggleGrid(){
  var state = $("input#grid").is(":checked");
  if (state == false) {
    map.removeLayer(grid);
  } else {
    map.addLayer(grid);
  }
}

// Option toggles
function toggleAntialias(){
  bathymetry.redraw();
  depth.redraw();
  grid.redraw();
}

// Bind controls
$("select#layer").on("change", updateLayer);
$("input#grid").on("change", toggleGrid);
$("input#depth").on("change", toggleDepth);
$("input#bathymetry").on("change", toggleBathymetry);
$("input#osm").on("change", toggleOsm);
$("input#antialias").on("change", toggleAntialias);

updateLayer();
toggleDepth();
toggleOsm();
