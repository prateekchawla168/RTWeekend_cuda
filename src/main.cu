
#include <stdio.h>          // printf, fprintf
#include <stdlib.h>         // abort

#include <iostream>
#include <fstream>
#include <time.h>


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Set up cuda and vulkan boilerplate
#include "cuda_setup.cuh"
#include "vulkan_setup.cuh"


void CUDARender() {

    // set up image
    int nx = 1280;
    int ny = 720;
    int numChannels = 3;
    
    // set up threadblocks - we can launch 32x32 threads at a time
    int tx = 8;
    int ty = 8;

    std::clog << "Rendering a " << nx << "x" << ny << " image ";
    std::clog << "in " << tx << "x" << ty << " blocks.\n";

    int numPixels = nx * ny;
    size_t fbSize = numChannels * numPixels * sizeof(float);

    // allocate fb
    float* fb;
    CheckCudaErrors(cudaMallocManaged((void**)&fb, fbSize));

    clock_t start, stop;

    start = clock();
    // render the buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    Render <<< blocks, threads >>> (fb, nx, ny);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());
    stop = clock();

    double timerSeconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::clog << "Render complete in " << timerSeconds << " sec.\n";


    // allocate ib
    unsigned char* ib;
    auto ibSize = numChannels * numPixels * sizeof(unsigned char);
    CheckCudaErrors(cudaMallocManaged((void**)&ib, ibSize));

    start = clock();
    ConvertFloatToInt <<< (numChannels*numPixels / 1024) , 1024 >>> (ib, fb, numChannels * numPixels);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());

    stop = clock(); 
    timerSeconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::clog << "Conversion to uint8 complete in " << timerSeconds << " sec.\n";

    stbi_write_png("output/HelloWorld.png", nx, ny, numChannels, ib, nx * numChannels);

    // remember to release memory!

    CheckCudaErrors(cudaFree(fb));
    CheckCudaErrors(cudaFree(ib));
    std::clog << "\nDone!\n";
}



// Main code
int main(){

    glfwSetErrorCallback(glfw_error_callback);
    if(!glfwInit())
        return 1;

    // create window with vulkan context
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1600, 1024, "Ray Tracing in a Weekend with CUDA and ImGui and Vulkan", nullptr, nullptr);
    if(!glfwVulkanSupported()){
        printf("[GLFW] Vulkan not supported!");
        return 1;
    }

    ImVector<const char*> extensions;
    uint32_t extensions_count = 0;
    const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&extensions_count);
    for (uint32_t i = 0; i < extensions_count; i++)
        extensions.push_back(glfw_extensions[i]);
    SetupVulkan(extensions);

    // Create window surface
    VkSurfaceKHR surface;
    VkResult err = glfwCreateWindowSurface(g_Instance, window, g_Allocator, &surface);
    check_vk_result(err);

    // create framebuffers
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    ImGui_ImplVulkanH_Window* wd = &g_MainWindowData;
    SetupVulkanWindow(wd, surface, w, h);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // enable keyboard

    //setup style
    ImGui::StyleColorsDark();

    // setup platform and renderer backends
    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = g_Instance;
    init_info.PhysicalDevice = g_PhysicalDevice;
    init_info.Device = g_Device;
    init_info.QueueFamily = g_QueueFamily;
    init_info.Queue = g_Queue;
    init_info.PipelineCache = g_PipelineCache;
    init_info.DescriptorPool = g_DescriptorPool;
    init_info.RenderPass = wd->RenderPass;
    init_info.Subpass = 0;
    init_info.MinImageCount = g_MinImageCount;
    init_info.ImageCount = wd->ImageCount;
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.Allocator = g_Allocator;
    init_info.CheckVkResultFn = check_vk_result;
    ImGui_ImplVulkan_Init(&init_info);

    // First, call the function containing the cuda render call
    CUDARender(); 

    // load our image as a texture
    ImageToTextureData tex_data;
    bool ret = LoadTextureFromFile("output/HelloWorld.png", &tex_data); 
    IM_ASSERT(ret);

    // our state
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    // main loop
    while(!glfwWindowShouldClose(window)){

        // poll and handle events
        glfwPollEvents();

        // resize swap chain?
        int fb_width, fb_height;
        glfwGetFramebufferSize(window, &fb_width, &fb_height);
        if (fb_width > 0 && fb_height > 0 && (g_SwapChainRebuild || g_MainWindowData.Width != fb_width || g_MainWindowData.Height != fb_height)){
            ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
            ImGui_ImplVulkanH_CreateOrResizeWindow(g_Instance, g_PhysicalDevice, g_Device, &g_MainWindowData, g_QueueFamily, g_Allocator, fb_width, fb_height, g_MinImageCount);
            g_MainWindowData.FrameIndex = 0;
            g_SwapChainRebuild = false;
        }
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED)!=0){
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        // start the dear imgui frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            // New Window
            ImGui::Begin("Render CUDA output with Vulkan");
        // just show the window we want
        {
            ImGui::BeginChild("Output:", ImVec2(0,0),ImGuiChildFlags_AlwaysAutoResize | ImGuiChildFlags_AutoResizeX | ImGuiChildFlags_AutoResizeY);
            ImGui::Image((ImTextureID)tex_data.DS, ImVec2(tex_data.width, tex_data.height) );
            ImGui::EndChild();
        }
        // small toolbox to show data
        {
            ImGui::BeginChild("Toolbox");
            ImGui::Text("Configuration options:");
            ImGui::ColorEdit4("Background color", (float*)&clear_color);
            ImGui::Text("Avg: %.1f fps", ImGui::GetIO().Framerate);
            ImGui::Text("Size = %d x %d", tex_data.width,  tex_data.height);
            ImGui::EndChild();
        }
            ImGui::End();
        }

        //Imgui render
        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();
        const bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);
        if (!is_minimized)
        {
            wd->ClearValue.color.float32[0] = clear_color.x * clear_color.w;
            wd->ClearValue.color.float32[1] = clear_color.y * clear_color.w;
            wd->ClearValue.color.float32[2] = clear_color.z * clear_color.w;
            wd->ClearValue.color.float32[3] = clear_color.w;
            FrameRender(wd, draw_data);
            FramePresent(wd);
        }

    }

    // cleanup
    err = vkDeviceWaitIdle(g_Device);
    check_vk_result(err);
    RemoveTexture(&tex_data);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    CleanupVulkanWindow();
    CleanupVulkan();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;


}