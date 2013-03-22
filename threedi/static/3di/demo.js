// Init the map
var map = new OpenLayers.Map(
  'map',
  {
    //projection: "EPSG:28992",
    projection: "EPSG:3857"
    //maxExtent: new OpenLayers.Bounds(12628.0541, 308179.0423, 283594.4779, 611063.1429)
  }
);
var depth = new OpenLayers.Layer.WMS(
  "Depth",
  "/3di/wms?layers=" + layer + "&time=" + time,
  {
    layers: "basic",
    transparent: "true"
  }
)
map.addLayer(depth);
var osm = new OpenLayers.Layer.OSM()
map.addLayer(osm);

// Functions
function prepare(prepare_type){
  var layer = $('select#layer option:selected').val()
  url = '/3di/wms?layer=' + layer + '&request=prepare';
  url = url + '&type=' + prepare_type;
  window.open(url)
}
function prepare_all() {prepare(null)}
function prepare_qm() {prepare('quad_monolith')}
function prepare_qp() {prepare('quad_pyramid')}
function prepare_hm() {prepare('height_monolith')}
function prepare_hp() {prepare('height_pyramid')}

function updateLayer(){
  var layer = $('select#layer option:selected')[0].getAttribute('value');
  // Determine bounds
  $.ajax(
    '/3di/wms',
    { 
      data: {
        request: 'getinfo',
        layer: layer,
        srs: 'epsg:3857'
      },
      success: updateLayerFromData
    }
  );
}

function updateLayerFromData(data) {
  $("#slider").slider("option", "max", data['timesteps'] - 1);
  var time = $("#slider").slider("value");
  updateTime(time);
  var bounds = data['bounds'];
  map.zoomToExtent(
    new OpenLayers.Bounds(bounds[0], bounds[1], bounds[2], bounds[3])
  )
}

function updateTime(time){
  $("#time").text(time)
  var layer = $('select#layer option:selected').val()
  depth.setUrl("/3di/wms?layer=" + layer + "&time=" + time);
  depth.redraw()
}

function toggleDepth(){
  var state = $("input#depth").is(":checked");
  depth.setVisibility(state);
}
function toggleOsm(){
  var state = $("input#osm").is(":checked");
  osm.setVisibility(state);
}

// Slider
function slide(ui, slider){
  updateTime(slider.value);
}

$("#slider").slider({
  min: 0,
  max: 143,
  slide: slide
});

// Bind controls
$("button#prepare-quad-monolith").on("click", prepare_qm);
$("button#prepare-quad-pyramid").on("click", prepare_qp);
$("button#prepare-height-monolith").on("click", prepare_hm);
$("button#prepare-height-pyramid").on("click", prepare_hp);
$("select#layer").on("change", updateLayer);
$("input#depth").on("change", toggleDepth);
$("input#osm").on("change", toggleOsm);

updateLayer();
