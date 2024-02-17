let map;
let styling =
    `<div>
        <h4 style="color: #491600; 
            font-family: Roboto; 
            font-size: 14px;
            font-style: normal;
            font-weight: 300;
            line-height: normal;
            margin-top: 0;">
            Location: Science Building
        </h4>
    </div>`;

//array of markers
let markers = [
    {
        coordinates: {lat:43.777714245931854, lng:-79.5025550471508},
        iconImage: "https://img.icons8.com/fluency/48/000000/marker-storm.png",
        content: styling
    },
    {
        coordinates: {lat:43.77170263276005, lng:-79.50474372951118},
        iconImage: "https://img.icons8.com/fluency/48/000000/marker-storm.png",
        content: styling
    }
]

// initialize map
function initMap() {
    const options = {
        zoom: 15,
        // coordinates
        center: {lat: 43.77408875774123, lng: -79.502254639768}
    }
    // map

    map = new google.maps.Map (
        document.getElementById('map'),
        options
    )

    //listen to map click
    // google.maps.event.addListener(map, "click", function(event) {
    //     //alert("aa")
    //     addMarker({
    //         coordinates: event.latLng
    //     })
    // })

    // styling = 
    //     `<div style="background-color:brown; 
    //             padding: 10px; 
    //             border-radius: 4px; 
    //             border-color: orange">
    //             <h4 style="color: white; 
    //                 font-family: Roboto; 
    //                 font-size: 20px;
    //                 font-style: normal;
    //                 font-weight: 400;
    //                 line-height: normal;
    //                 margin-top: 0;
    //                 ">
    //                 Location: Science Building
    //             </h4>
    //         </div>`
    

    //let marker = new google.maps.Marker({ 
        position: //{lat:43.777714245931854, lng:-79.5025550471508},
        map: //map,
        //icon: "https://img.icons8.com/fluent/48/000000/marker-storm.png" https://icons8.com/icon/E9btaaI7vsaa/marker-storm
      // })
    
    //array of markers
    // let markers = [
    //     {
    //         coordinates: {lat:43.777714245931854, lng:-79.5025550471508},
    //         iconImage: "https://img.icons8.com/fluency/48/000000/marker-storm.png",
    //         content: styling
    //     },
    //     {
    //         coordinates: {lat:43.77170263276005, lng:-79.50474372951118},
    //         iconImage: "https://img.icons8.com/fluency/48/000000/marker-storm.png",
    //         content: styling
    //     }
    // ]

    for (let i=0; i < markers.length; i++) {
        addMarker(markers[i])
    }

    // addMarker(
    //     {
    //         coordinates: {lat:43.777714245931854, lng:-79.5025550471508},
    //         iconImage: "https://img.icons8.com/fluency/48/000000/marker-storm.png",
    //         content: styling
    //     }
    // )
    //alert('initMap'); alert user initMap
}

function addMarker(prop) {
    let marker = new google.maps.Marker({ 
        position: prop.coordinates,
        map: map,
        icon: prop.iconImage

    })

    //if(prop.iconImage) {
        //marker.setIcon(prop.iconImage)
    //}

    if(prop.content) {
        let info = new google.maps.InfoWindow({
            content: prop.content
        })

        marker.addListener("click", function() {
            info.open(map, marker)
        })
    }
}
