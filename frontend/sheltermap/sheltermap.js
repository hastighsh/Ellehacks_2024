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

// let products = [
//     { imgSrc: 'img/b1.png', name: 'Benefit Cosmetics - Hoola Bronzer', price: '$47.00' },
//     { imgSrc: 'img/b2.png', name: 'Too Faced - Chocolate Soleil Matte Bronzer', price: '$48.50' },
//     { imgSrc: 'img/b3.png', name: 'Rare Beauty by Selena Gomez - Warm Wishes Effortless Bronzer Sticks', price: '$34.00' },
//     { imgSrc: 'img/c1.png', name: 'NARS - Radiant Creamy Concealer', price: '$42.00' },
//     { imgSrc: 'img/c2.png', name: 'Dior - Backstage Concealer', price: '$38.00' },
//     { imgSrc: 'img/g1.png', name: 'Dior - Lip Glow Oil', price: '$50.00' },
//     { imgSrc: 'img/g2.png', name: 'Summer Fridays - Lip Butter Balm', price: '$31.00' },
//     { imgSrc: 'img/g3.png', name: 'fresh - Sugar Lip Balm Hydrating Treatment', price: '$33.00' },
//     { imgSrc: 'img/m1.png', name: 'Drunk Elephant - Protini Polypeptide Firming Refillable Moisturizer', price: '$89.00' },
//     { imgSrc: 'img/m2.png', name: 'Tatcha - The Dewy Skin Cream Plumping and Hydrating Moisturizer', price: '$97.00' },
//     { imgSrc: 'img/m3.png', name: 'Glow Recipe - Plum Plump Refillable Hyaluronic Acid Moisturizer', price: '$52.00' }
// ];

// // Get the product list container
// const productList = document.getElementById('product-list');

// // Loop through the products array and generate HTML for each product
// for (let i = 0; i < products.length; i++) {
//     const product = products[i];

//     // Create a new div element for the product
//     const productDiv = document.createElement('div');
//     productDiv.classList.add('product');

//     // Create an image element for the product
//     const img = document.createElement('img');
//     img.src = product.imgSrc;
//     img.alt = '';

//     // Create a div element for the product details
//     const detailsDiv = document.createElement('div');
//     detailsDiv.classList.add('p-details');

//     // Create h2 and h3 elements for the product name and price
//     const nameHeading = document.createElement('h2');
//     nameHeading.textContent = product.name;
//     const priceHeading = document.createElement('h3');
//     priceHeading.textContent = product.price;

//     // Append the image and details to the product div
//     detailsDiv.appendChild(nameHeading);
//     detailsDiv.appendChild(priceHeading);
//     productDiv.appendChild(img);
//     productDiv.appendChild(detailsDiv);

//     // Append the product div to the product list container
//     productList.appendChild(productDiv);
// }

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



// function addProduct(prop) {
//     // Get the product list container
//     const productList = document.getElementById('product-list');

//     // Loop through the products array and generate HTML for each product
//     for (let i = 0; i < products.length; i++) {
//         const product = products[i];

//         // Create a new div element for the product
//         const productDiv = document.createElement('div');
//         productDiv.classList.add('product');

//         // Create an image element for the product
//         const img = document.createElement('img');
//         img.src = product.imgSrc;
//         img.alt = '';

//         // Create a div element for the product details
//         const detailsDiv = document.createElement('div');
//         detailsDiv.classList.add('p-details');

//         // Create h2 and h3 elements for the product name and price
//         const nameHeading = document.createElement('h2');
//         nameHeading.textContent = product.name;
//         const priceHeading = document.createElement('h3');
//         priceHeading.textContent = product.price;

//         // Append the image and details to the product div
//         detailsDiv.appendChild(nameHeading);
//         detailsDiv.appendChild(priceHeading);
//         productDiv.appendChild(img);
//         productDiv.appendChild(detailsDiv);

//     // Append the product div to the product list container
//         productList.appendChild(productDiv);
//     }
// }

// const search = () => {
//     // document.getElementById("id name in html")
//     //.value takes the user input
//     const searchbox = document.getElementById("search-item").value.toUpperCase();
//     const storeitems = document.getElementById("product-list");
//     // storeitems represents the container holding all the product items
//     const product = document.querySelectorAll(".product");
//     // product stores all the items with the class product
//     const pname = storeitems.getElementsByTagName("h2");
//     // gets all the storeitems elements and pname holds all the h2 elements

//     for (var i = 0; i < pname.length; i++) {
//         let match = product[i].getElementsByTagName("h2")[0];
//         // takes whatever iteration we are on's product and the name under h2 and [0] makes sure it's the first name element in case there are multiples
        
//         if (match) {
//             // checked if match has a value
//             let textvalue = match.textContent || match.innerHTML;
//             // match.textContent retrieves the text content of the h2 element stores in match
//             // if .textContent is not supported it falls on match.innerHTML
//             if (textvalue.toUpperCase().indexOf(searchbox) > -1) {
//                 // text content is made case insensitive and the searchbox value index in the text content is found if it exists it won't be -1
//                 product[i].style.display = "";
//                 // when set to an empty string, the product is displayed
//             } else {
//                 product[i].style.display = "none";
//             }
//         }
//     }
// }

// // initialize map
// function initMap() {
//     const options = {
//         zoom: 15,
//         // coordinates
//         center: {lat: 43.77408875774123, lng: -79.502254639768}
//     }
//     // map

//     map = new google.maps.Map (
//         document.getElementById('map'),
//         options
//     )

//     //listen to map click
//     // google.maps.event.addListener(map, "click", function(event) {
//     //     //alert("aa")
//     //     addMarker({
//     //         coordinates: event.latLng
//     //     })
//     // })

//     // styling = 
//     //     `<div style="background-color:brown; 
//     //             padding: 10px; 
//     //             border-radius: 4px; 
//     //             border-color: orange">
//     //             <h4 style="color: white; 
//     //                 font-family: Roboto; 
//     //                 font-size: 20px;
//     //                 font-style: normal;
//     //                 font-weight: 400;
//     //                 line-height: normal;
//     //                 margin-top: 0;
//     //                 ">
//     //                 Location: Science Building
//     //             </h4>
//     //         </div>`
    

//     //let marker = new google.maps.Marker({ 
//         position: //{lat:43.777714245931854, lng:-79.5025550471508},
//         map: //map,
//         //icon: "https://img.icons8.com/fluent/48/000000/marker-storm.png" https://icons8.com/icon/E9btaaI7vsaa/marker-storm
//       // })
    
//     //array of markers
//     // let markers = [
//     //     {
//     //         coordinates: {lat:43.777714245931854, lng:-79.5025550471508},
//     //         iconImage: "https://img.icons8.com/fluency/48/000000/marker-storm.png",
//     //         content: styling
//     //     },
//     //     {
//     //         coordinates: {lat:43.77170263276005, lng:-79.50474372951118},
//     //         iconImage: "https://img.icons8.com/fluency/48/000000/marker-storm.png",
//     //         content: styling
//     //     }
//     // ]

//     for (let i=0; i < markers.length; i++) {
//         addMarker(markers[i])
//     }

//     // addMarker(
//     //     {
//     //         coordinates: {lat:43.777714245931854, lng:-79.5025550471508},
//     //         iconImage: "https://img.icons8.com/fluency/48/000000/marker-storm.png",
//     //         content: styling
//     //     }
//     // )
//     //alert('initMap'); alert user initMap
// }

// .container {
//     /*this code block is to create the big white rectangle*/
//     position: fixed;
//     right: 350px;
//     bottom: 70px;
//     width: 500px;
//     height: 600px;
//     background: #fff;
//     /*border-radius gives a curved corner border with a radius of 15px*/
//     border-radius: 15px;
//     /*box-shadow: x-coordinate y-coordinate radius rgba(red, green, blue, alpha)*/
//     box-shadow: 4px 4px 30px rgba(0, 0, 0, 0.06);
//     padding: 20px;
//     /*when set to "scroll" it enables verical scrolling within the container if its content overflows*/
//     overflow-y: scroll;
// }

// .container::-webkit-scrollbar {
//     /*hides the scrollbar*/
//     display: none;
// }

// .container form {
//     /*the space "form" refers to the nested html code container->form*/
//     width: 100%;
//     border: 1px solid;
//     border-radius: 4px;
//     /*code below makes sure the input is centered in the form*/
//     display: flex;
//     flex-direction: row;
//     align-items: center;
// }

// /*styles the search input field*/
// .container form input {
//     /*set to none to remove any default styling*/
//     border: none;
//     outline: none;
//     box-shadow: none;
//     width: 100%;
//     font-size: 16px;
//     font-weight: 500;
//     padding: 8px 10px; 
//     /*padding: topBtm leftRight;*/
// }

// #product-list  {
//     padding: 20px 0;
// }

// .product {
//     /*display and align-items horizontally align the product image and details with the product item*/
//     display: flex;
//     align-items: center;
//     cursor: pointer;
//     /*cursor code changes to a mickeymouse hand when over product with pointer*/
//     padding-bottom: 15px;
// }

// .product img {
//     width: 70px;
//     height: 70px;
//     /*maintains image proportions*/
//     object-fit: cover;
//     border-radius: 5px;
// }

// .product .p-details {
//     padding-left: 15px;
// }

// .product .p-details .long {
//     font-size: 14px;
//     font-weight: 400;
//     color: #1d1d1d;
// }

// .product .p-details h2 {
//     /*classes need the period and <title> don't*/
//     font-size: 20px;
//     font-weight: 400;
//     color: #1d1d1d;
// }

// .product .p-details h3 {
//     font-size: 18px;
//     font-weight: 600;
// }